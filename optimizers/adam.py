import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
from optimizers.optimizer_utils import Optimizer
import jax.numpy as jnp

import wandb
import matplotlib.pyplot as plt

def make_adam(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        use_ema: bool = False,
        ema_rate: float = 0.999,
        ema_bias_correction: bool = True,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        variance = otu.tree_zeros_like(params)
        if ema_bias_correction:
            ema = otu.tree_zeros_like(params)
        else:
            ema = params
        step = 0
        return (momentum, variance, ema, step)
    
    def update_fn(grads, state, params):
        momentum, variance, ema, step = state
        step = step + 1
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        variance = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, variance, grads)
        momentum_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum)
        variance_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance)
        updates = jax.tree_map(lambda m, v: m / (v ** 0.5 + eps), momentum_hat, variance_hat)
        ema = jax.tree_map(lambda e, p: ema_rate * e + (1 - ema_rate) * p, ema, params)
        return updates, (momentum, variance, ema, step)
    
    def get_grad_params(state, params):
        return params

    @jax.jit
    def get_eval_params(state, params):
        if not use_ema:
            return params
        _, _, ema, step = state
        if ema_bias_correction:
            return jax.tree_map(lambda e: e / (1 - ema_rate ** step), ema)
        else:
            return ema

    def log_stats(state, grads, params, log_step):
        look = lambda x: x['Block_3']['Dense_0']['kernel']
        momentum, variance, ema, step = state
        momentum = jnp.reshape(look(momentum), -1)
        variance = jnp.reshape(look(variance), -1)
        grads = jnp.reshape(look(grads), -1)
        params = jnp.reshape(look(params), -1)
        ema = jnp.reshape(look(ema), -1)
        momentum_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum)
        variance_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance)
        updates = jnp.reshape(jax.tree_map(lambda m, v: m / (v ** 0.5 + eps), momentum_hat, variance_hat))

        info = {
            'momentum_max': jnp.max(jnp.abs(momentum)),
            'momentum_mean': jnp.mean(jnp.abs(momentum)),
            'variance_max': jnp.max(jnp.abs(variance)),
            'variance_mean': jnp.mean(jnp.abs(variance)),
            'variance_min': jnp.min(jnp.abs(variance)),
            'ema_max': jnp.max(jnp.abs(ema)),
            'ema_mean': jnp.mean(jnp.abs(ema)),
            'grads_max': jnp.max(jnp.abs(grads)),
            'grads_mean': jnp.mean(jnp.abs(grads)),
            'params_max': jnp.max(jnp.abs(params)),
            'params_mean': jnp.mean(jnp.abs(params)),
            'updates_max': jnp.max(jnp.abs(updates)),
            'updates_mean': jnp.mean(jnp.abs(updates)),
        }
        wandb.log({'optimizer/'+k : v for k,v in info.items()}, step=log_step)

        def log_histogram(vec, name):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(vec, bins=100, color='skyblue', alpha=0.7)
            ax.set_yscale('log')
            wandb.log({'optimizer_imgs/'+name: wandb.Image(fig)}, step=log_step)
            plt.close(fig)
            return fig
        log_histogram(momentum, 'momentum_histogram')
        log_histogram(variance, 'variance_histogram')
        log_histogram(grads, 'grads_histogram')
        log_histogram(params, 'params_histogram')
        log_histogram(updates, 'updates_histogram')
        
    
    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, None, log_stats)
