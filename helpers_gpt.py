import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import tiktoken
import flax.linen as nn

enc = tiktoken.get_encoding('gpt2')

def log_real_data(batch, step):
    batch_local = jax.experimental.multihost_utils.process_allgather(batch)

@jax.jit
def call_model(train_state, x):
    return train_state.call_model(x.astype(jnp.uint32), None, None)

def log_generated_data(FLAGS, train_state, shard_data, batch, step):
    batch_local, _ = jax.experimental.multihost_utils.process_allgather(batch)

    # Real Data
    if jax.process_index() == 0:
        decoded_text = [enc.decode(tokens) for tokens in batch_local[:8]]
        table = wandb.Table(columns=['Real Text'], data=[[t] for t in decoded_text])
        wandb.log({"real_text": table}, step=step)

    # Generated Data
    with jax.spmd_mode('allow_all'):
        temperature = 0.1
        key = jax.random.PRNGKey(0)
        key = jax.random.fold_in(key, jax.process_index())
        tokens = batch_local[:, :10]
        for i in range(256-10):
            key, eps_key = jax.random.split(key)
            tokens_padded = jnp.pad(tokens, ((0,0), (0, 256-tokens.shape[1])), constant_values=0)
            logits = call_model(train_state, tokens_padded) / temperature
            next_tokens_all = jnp.argmax(logits, axis=-1)
            next_tokens = next_tokens_all[:, tokens.shape[1]-1]

            tokens = jnp.concatenate([tokens, next_tokens[:, None]], axis=-1)
        tokens = jax.experimental.multihost_utils.process_allgather(tokens)

        if jax.process_index() == 0:
            decoded_text = [f'[{enc.decode(batch_local[i, :10])}]'+enc.decode(tokens[i]) for i in range(tokens.shape[0])]
            table = wandb.Table(columns=['Generated Text'], data=[[t] for t in decoded_text])
            wandb.log({"generated_text": table}, step=step)

            generated_tokens = tokens
            real_tokens = batch_local[:, :generated_tokens.shape[1]]
            wandb.log({"online_dataset_accuracy": jnp.mean(real_tokens == tokens)}, step=step)