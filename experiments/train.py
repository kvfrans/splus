import jax.experimental
import jax.experimental.multihost_utils
from localutils.debugger import enable_debug
enable_debug()

from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections
import matplotlib.pyplot as plt

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache('/nfs/jax-cache')

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from helpers_logging import create_horizontal_bars
from model import Transformer

from optimizers.adam import make_adam
from optimizers.schedule_free import make_schedule_free
from optimizers.soap import make_soap
from optimizers.psgd import make_psgd
from optimizers.shampoo import make_shampoo
from optimizers.sophia import make_sophia
from optimizers.muon import make_muon
from optimizers.psgd import make_psgd
from optimizers.splus import make_splus
from optimizers.soap_custom import make_soap_custom

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_bool('debug_overfit', False, 'Flag to overfit on small data.')
flags.DEFINE_string('label', None, 'Label for wandb run.')


model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'lr_embed_first': -1.0,
    'lr_embed_last': -1.0,
    'beta1': 0.9,
    'beta2': 0.95,
    'ema_rate': 0.999,
    'weight_decay': 0.1,
    'warmup': 10_000,
    'hidden_size': 64, # change this!
    'patch_size': 8, # change this!
    'depth': 2, # change this!
    'num_heads': 2, # change this!
    'mlp_ratio': 1, # change this!
    'sharding': 'dp', # dp or fsdp.
    'train_type': 'dit', # 'dit', 'vit', 'gpt'
    # Optimizer options
    'optimizer': 'adam',
    'optimizer_freq': 100,
    'splus_constant': 0.001,
    'do_scaling': 1,
    'scaling_type': 'first',
    # Diffusion options
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 1.5,
    'use_stable_vae': 1,
    # GPT options
    'sequence_length': 256,
    'vocab_size': 50257 # Don't change if using default tiktoken
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'batchsize',
    'name': 'batchsize_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)
    
##############################################
## Training Code.
##############################################
def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        FLAGS.wandb.name = (FLAGS.wandb.name + '-' + FLAGS.label) if FLAGS.label is not None else FLAGS.wandb.name
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)
        
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, is_train=True, max_sequence_length=FLAGS.model.sequence_length, debug_overfit=FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, 32, is_train=False, max_sequence_length=FLAGS.model.sequence_length, debug_overfit=FLAGS.debug_overfit)
    example_data, example_label = next(dataset)
    if FLAGS.model.train_type == 'gpt':
        example_data = example_data[:, 1:]

    if FLAGS.model.use_stable_vae:
        assert FLAGS.model.train_type != 'gpt' # VAE should only be used for images.
        vae = StableVAE.create()
        vae_decode = jax.jit(vae.decode)
        x_shape = vae.encode(jax.random.PRNGKey(0), example_data[0:1]).shape[1:]
    else:
        x_shape = example_data.shape[1:]

    if FLAGS.fid_stats is not None:
        truth_fid_stats = np.load(FLAGS.fid_stats)

    ###################################
    # Creating Model and put on devices.
    ###################################
    transformer_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'num_classes': FLAGS.model['num_classes'] if FLAGS.model['train_type'] != 'gpt' else FLAGS.model['vocab_size'],
        'train_type': FLAGS.model['train_type'],
    }
    model_def = Transformer(**transformer_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    placeholder_input = (example_data, jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))
    placeholder_params = jax.eval_shape(model_def.init, jax.random.PRNGKey(0), *placeholder_input)['params']
    print(tabulate_fn(*placeholder_input))
    def do_weight_decay_mask(path, param):
        full_path = '/'.join([p.key for p in path]).lower()
        if 'bias' in full_path or 'embed' in full_path or 'layernorm' in full_path:
            return False
        return True
    weight_decay_mask = jax.tree_util.tree_map_with_path(do_weight_decay_mask, placeholder_params)
    lr_schedule = optax.linear_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])

    if FLAGS.model.optimizer == 'adam':
        optimizer = make_adam(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    elif FLAGS.model.optimizer == 'schedule_free':
        optimizer = make_schedule_free(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    elif FLAGS.model.optimizer == 'soap':
        optimizer = make_soap(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    elif FLAGS.model.optimizer == 'psgd':
        optimizer = make_psgd()
    elif FLAGS.model.optimizer == 'muon':
        optimizer = make_muon(beta=FLAGS.model['beta1'])
    elif FLAGS.model.optimizer == 'shampoo':
        optimizer = make_shampoo(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    elif FLAGS.model.optimizer == 'sophia':
        optimizer = make_sophia(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'])
    elif FLAGS.model.optimizer == 'splus':
        optimizer = make_splus(b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], ema_rate=FLAGS.model['ema_rate'])
    optim_components = [
        optimizer,
        optax.add_decayed_weights(FLAGS.model['weight_decay'], mask=weight_decay_mask),
        optax.scale_by_learning_rate(lambda x: lr_schedule(x)),
    ]
    tx = optax.chain(*optim_components)

    def init(rng):
        param_key, dropout_key = jax.random.split(rng, 2)
        model_rngs = {'params': param_key, 'dropout': dropout_key}
        params = model_def.init(model_rngs, jnp.zeros((1, *x_shape)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32))['params']

        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    start_step = 0


    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        train_state = train_state.replace(params=replace_dict['params'], params_avg=replace_dict['params']) # Only load params.
        train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", replace_dict['step'])
        train_state = train_state.replace(step=0)
        del cp

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding.params, no_shard), static_argnames=('return_activations', 'on_policy'))
    def get_grads(params, batch, key, return_activations=False, on_policy=False):
        def loss_fn_dit(grad_params):
            info = {}
            images, labels = batch
            t_key, eps_key, labels_key, vae_key = jax.random.split(key, 4)
            if FLAGS.model.use_stable_vae:
                images = vae.encode(vae_key, images)
            labels_dropout = jax.random.bernoulli(labels_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
            labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
            info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])
            t = jax.random.randint(t_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
            t /= FLAGS.model['denoise_timesteps']
            t_full = t[:, None, None, None] # [batch, 1, 1, 1]
            x_1 = images
            x_0 = jax.random.normal(eps_key, images.shape)
            x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
            v_t = x_1 - (1 - 1e-5) * x_0
            v_prime, activations, infos = train_state.call_model(x_t, t, labels_dropped, params=grad_params, return_activations=True)
            loss = jnp.mean((v_prime - v_t) ** 2)
            info['v_magnitude_prime'] = jnp.sqrt(jnp.mean(jnp.square(v_prime)))
            info['loss'] = loss
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)
        def loss_fn_vit(grad_params):
            info = {}
            images, labels = batch
            if FLAGS.model.use_stable_vae:
                images = vae.encode(key, images)
            logits, activations, infos = train_state.call_model(images, None, None, params=grad_params, return_activations=True)
            log_probs = jax.nn.log_softmax(logits)
            loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(labels, FLAGS.model['num_classes']), axis=-1))
            info['loss'] = loss
            info['accuracy'] = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)
        def loss_fn_gpt(grad_params):
            info = {}
            text, _ = batch
            text_input, text_target = text[:, :-1], text[:, 1:]
            logits, activations, infos = train_state.call_model(text_input, None, None, params=grad_params, return_activations=True)
            log_probs = jax.nn.log_softmax(logits)
            if on_policy:
                text_target = jax.random.categorical(train_state.rng, logits)
            loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(text_target, FLAGS.model['vocab_size']), axis=-1))
            info['loss'] = loss
            info['accuracy'] = jnp.mean(jnp.argmax(logits, axis=-1) == text_target)
            info.update({'infos/'+k:v for k, v in infos.items()})
            return loss, (activations, info)

        loss_fn = {'dit': loss_fn_dit, 'vit': loss_fn_vit, 'gpt': loss_fn_gpt}[FLAGS.model.train_type]
        grads, (activations, info) = jax.grad(loss_fn, has_aux=True)(params)

         # Log some statistics about activations, params, etc.
        info.update({'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()})
        if return_activations:
            info.update({'activations_full/' + k : v for k, v in activations.items()})

        info['grad_max'] = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(jnp.max(x), jnp.max(y)), grads)
        info['grad_norm'] = optax.global_norm(grads)
        info['param_max'] = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(jnp.max(x), jnp.max(y)), params)
        info['param_norm'] = optax.global_norm(params)
        return grads, info

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update_fn(train_state, grads):
        info = {}
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        info['lr'] = lr_schedule(train_state.step)

        if FLAGS.model.optimizer == 'splus':
            def shape_scale(path, u):
                path_str = '/'.join([p.key for p in path])
                if FLAGS.model['do_scaling']:
                    if len(u.shape) == 2 and not ('TokenEmbedder' in path_str or 'ClassifierOutput' in path_str or 'TimestepEmbedder' in path_str):
                        if FLAGS.model['scaling_type'] == 'first':
                            scale = 1 / u.shape[0]
                        elif FLAGS.model['scaling_type'] == 'avg':
                            scale = 1 / ((u.shape[0] + u.shape[1])/2)
                        else:
                            raise NotImplementedError
                    else:
                        scale = FLAGS.model['splus_constant']
                else:
                    scale = 1
                print(f"Scaling {u.shape} with {scale}; total is {FLAGS.model['lr'] * scale}")
                return u * scale
            updates = jax.tree_util.tree_map_with_path(shape_scale, updates)

        new_params = optax.apply_updates(train_state.params, updates)
        info['update_norm'] = optax.global_norm(updates)
        info['update_max'] = jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))), updates)
        train_state = train_state.replace(step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_avg()
        return train_state, info
    
    ###################################
    # Train Loop
    ###################################
    
    rng_key = jax.random.PRNGKey(0)
    valid_batch = shard_data(*next(dataset_valid))
    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        rng_key, rng = jax.random.split(rng_key)
        # Update.
        if i == 1 or not FLAGS.debug_overfit:
            batch = shard_data(*next(dataset))

        grad_params = optimizer.get_grad_params(train_state.opt_state[0], train_state.params)
        grads, grads_info = get_grads(grad_params, batch, rng)
        train_state, update_info = update_fn(train_state, grads)

        if (i == 1 or i % FLAGS.model['optimizer_freq'] == 0) and optimizer.update_slow is not None:
            del grads, grad_params
            new_opt_state = optimizer.update_slow(train_state.opt_state[0], train_state_sharding.opt_state[0], train_state=train_state, batch=shard_data(*next(dataset)))
            train_state = train_state.replace(opt_state=(new_opt_state, *train_state.opt_state[1:]))
        update_info = {**grads_info, **update_info}

        # Per-update logs.
        if i % FLAGS.log_interval == 0 or (i < 10):
            update_info = jax.device_get(update_info)
            update_info = jax.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            if not FLAGS.debug_overfit:
                eval_params = optimizer.get_eval_params(train_state.opt_state[0], train_state.params)
                _, valid_update_info = get_grads(eval_params, valid_batch, rng)
                valid_update_info = jax.device_get(valid_update_info)
                valid_update_info = jax.tree_map(lambda x: x.mean(), valid_update_info)
                train_metrics['training/loss_valid'] = valid_update_info['loss']
                train_metrics['training/accuracy_valid'] = valid_update_info.get('accuracy', 0.0)

            if jax.process_index() == 0:
                print(train_metrics['training/loss_valid'])
                wandb.log(train_metrics, step=i)

        if (i % FLAGS.save_interval == 0 or i in [1, 1000]) and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(FLAGS.save_dir+'-step'+str(train_state_gather.step), parallel=False)
                cp.train_state = train_state_gather
                cp.save()
                del cp
            del train_state_gather

if __name__ == '__main__':
    app.run(main)