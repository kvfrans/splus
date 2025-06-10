import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt
import flax.linen as nn
import sys
import tiktoken
from splus import splus, splus_get_eval_params

from localutils.debugger import enable_debug
enable_debug()

from jaxtransformer.utils.wandb import setup_wandb
from jaxtransformer.utils.datasets import get_dataset
from jaxtransformer.configs import training_config, optimizer_config, transformer_config, wandb_config
from jaxtransformer.utils.train_state import TrainState
from jaxtransformer.utils.checkpoint import Checkpoint
from jaxtransformer.utils.sharding import create_sharding
from jaxtransformer.transformer import TransformerBackbone
from jaxtransformer.modalities import TokenEmbed, ClassifierOutput

llm_config = ml_collections.ConfigDict({
    'sequence_length': 1024,
    'vocab_size': 50257, # Default tiktoken vocab size.
})
training_config['log_interval'] = 10
config_flags.DEFINE_config_dict('train', training_config, lock_config=False)
config_flags.DEFINE_config_dict('tf', transformer_config, lock_config=False)
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('llm', llm_config, lock_config=False)
FLAGS = flags.FLAGS

flags.DEFINE_string('optimizer', 'splus', 'Either adam or splus.')
flags.DEFINE_float('lr', 0.3, 'Learning rate.')
FLAGS(sys.argv)

###################################
# Model Definition.
###################################
class LLMModel(nn.Module):
    @nn.compact
    def __call__(self, seq_ids):
        seq_ids = seq_ids.astype(jnp.uint32)
        x = TokenEmbed(FLAGS.llm.vocab_size, FLAGS.tf.hidden_size)(seq_ids)
        pos_embed = TokenEmbed(x.shape[1], FLAGS.tf.hidden_size)(jnp.arange(x.shape[1])[None, :])
        x = x + pos_embed
        x = TransformerBackbone(**FLAGS.tf, use_conditioning=False, use_causal_masking=True)(x)
        x = ClassifierOutput(FLAGS.llm.vocab_size)(x)
        return x
    
    def weight_decay_mask(self, params): # No clean way to define this in __call__, unfortunately.
        weight_decay_mask = {p: True for p in params.keys()}
        weight_decay_mask['TokenEmbed_0'] = False
        weight_decay_mask['TokenEmbed_1'] = False
        return weight_decay_mask
    
##############################################
## Initialization.
##############################################

if jax.process_index() == 0:
    setup_wandb(FLAGS.flag_values_dict(), **FLAGS.wandb)
np.random.seed(FLAGS.train.seed)
rng = jax.random.PRNGKey(FLAGS.train.seed)
# if FLAGS.train.use_jit_cache:
#     from jax.experimental.compilation_cache import compilation_cache as cc
#     cc.initialize_cache('/home/kvfrans/jax-cache')

# Load dataset and helper modules.
enc = tiktoken.get_encoding('gpt2')
dataset, dataset_valid = [get_dataset(FLAGS.train.dataset_name, FLAGS.train.batch_size, is_train=is_train, max_sequence_length=FLAGS.llm.sequence_length) for is_train in (True, False)]
example_text, _ = next(dataset)

# Initialize model, parameters, optimizer, train state.
model_def = LLMModel()
placeholder_input = (example_text[:, :-1],)
placeholder_params = jax.eval_shape(model_def.init, jax.random.PRNGKey(0), *placeholder_input)['params']
weight_decay_mask = model_def.weight_decay_mask(placeholder_params)
lr_schedule = optax.linear_schedule(0.0, FLAGS.lr, 200)
if FLAGS.optimizer == 'adam':
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001, mask=weight_decay_mask)
elif FLAGS.optimizer == 'splus':
    tx = splus(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001, mask=weight_decay_mask)
    # tx = adam(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001, mask=weight_decay_mask)
            
init_fn = partial(TrainState.create, model_def=model_def, model_input=placeholder_input, tx=tx)
train_state_shape = jax.eval_shape(init_fn, rng=rng)
train_state_sharding, no_sharding, shard_data = create_sharding(FLAGS.train.sharding, train_state_shape)
train_state = jax.jit(init_fn, out_shardings=train_state_sharding)(rng=rng)
start_step = 0
print(nn.tabulate(model_def, jax.random.PRNGKey(0))(*placeholder_input))

if FLAGS.train.load_dir is not None:
    cp = Checkpoint(FLAGS.train.load_dir)
    train_state = train_state.replace(**cp.load_as_dict()['train_state'])
    train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
    print("Loaded model with step", train_state.step)
    train_state = train_state.replace(step=0)
    del cp

###################################
# Update Function
###################################

@partial(jax.jit, out_shardings=(train_state_sharding, no_sharding))
def update(train_state, batch):
    new_rng, key = jax.random.split(train_state.rng)
    text, _ = batch
    text_input, text_target = text[:, :-1], text[:, 1:]
    def loss_fn(grad_params):
        logits = train_state.call_model(text_input, params=grad_params)
        log_probs = jax.nn.log_softmax(logits)
        loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(text_target, FLAGS.llm.vocab_size), axis=-1))
        return loss, {
            'loss': loss,
            'accuracy': jnp.mean(jnp.argmax(logits, axis=-1) == text_target),
            'perplexity': jnp.exp(loss)
        }
    
    grads, info = jax.grad(loss_fn, has_aux=True)(train_state.params)
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    info['grad_norm'] = optax.global_norm(grads)
    info['update_norm'] = optax.global_norm(updates)
    info['param_norm'] = optax.global_norm(new_params)
    info['lr'] = lr_schedule(train_state.step)

    train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
    return train_state, info

@jax.jit
def call_model(train_state, x):
    return train_state.call_model(x.astype(jnp.uint32))

###################################
# Train Loop
###################################

for i in tqdm.tqdm(range(1, FLAGS.train.max_steps + 1),
                    smoothing=0.1,
                    dynamic_ncols=True):
    
    # Update.
    batch = shard_data(*next(dataset))
    train_state, update_info = update(train_state, batch)

    # Per-update logs.
    if i % FLAGS.train.log_interval == 0:
        update_info = jax.device_get(update_info)
        update_info = jax.tree_map(lambda x: np.array(x), update_info)
        update_info = jax.tree_map(lambda x: x.mean(), update_info)
        train_metrics = {f'training/{k}': v for k, v in update_info.items()}

        valid_batch = shard_data(*next(dataset_valid))
        if FLAGS.optimizer == 'splus':
            train_state_eval = train_state.replace(params=splus_get_eval_params(train_state.opt_state[0]))
        else:
            train_state_eval = train_state
        _, valid_update_info = update(train_state_eval, valid_batch)
        valid_update_info = jax.device_get(valid_update_info)
        valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
        train_metrics['training/loss_valid'] = valid_update_info['loss']
        train_metrics['training/accuracy_valid'] = valid_update_info['accuracy']
        train_metrics['training/perplexity_valid'] = valid_update_info['perplexity']

        if jax.process_index() == 0:
            wandb.log(train_metrics, step=i)

    # if i % FLAGS.train.eval_interval == 0 or i in (0, 1000, 10000):
    #     batch_local, _ = jax.experimental.multihost_utils.process_allgather(batch)
    #     batch_local = batch_local[:8]
    #     with jax.spmd_mode('allow_all'):
    #         temperature = 0.7
    #         key = jax.random.PRNGKey(0)
    #         key = jax.random.fold_in(key, jax.process_index())
    #         tokens = batch_local[:, :10]
    #         for _ in range(FLAGS.llm.sequence_length-10):
    #             key, eps_key = jax.random.split(key)
    #             tokens_padded = jnp.pad(tokens, ((0,0), (0, FLAGS.llm.sequence_length-tokens.shape[1])), constant_values=0)
    #             logits = call_model(train_state, tokens_padded) / temperature
    #             next_logits = logits[:, tokens.shape[1]-1]
    #             next_tokens = jax.random.categorical(eps_key, next_logits)

    #             tokens = jnp.concatenate([tokens, next_tokens[:, None]], axis=-1)
    #         tokens = jax.experimental.multihost_utils.process_allgather(tokens)

    #         if jax.process_index() == 0:
    #             decoded_text = [f'[{enc.decode(batch_local[i, :10])}]'+enc.decode(tokens[i]) for i in range(tokens.shape[0])]
    #             table = wandb.Table(columns=['Generated Text'], data=[[t] for t in decoded_text])
    #             wandb.log({"generated_text": table}, step=i)

    #             decoded_text = [enc.decode(tokens) for tokens in batch_local[:8]]
    #             table = wandb.Table(columns=['Real Text'], data=[[t] for t in decoded_text])
    #             wandb.log({"real_text": table}, step=i)

    #             generated_tokens = tokens
    #             real_tokens = batch_local[:, :generated_tokens.shape[1]]
    #             wandb.log({"online_dataset_accuracy": jnp.mean(real_tokens == tokens)}, step=i)