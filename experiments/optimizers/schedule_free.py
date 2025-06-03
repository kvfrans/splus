import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
import jax.numpy as jnp
from optimizers.optimizer_utils import Optimizer

def make_schedule_free(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        use_polyak: bool = True,
        ema_rate: float = 0.999,
        ema_bias_correction: bool = True,
    ):

    def init_fn(params):
        variance = otu.tree_zeros_like(params)
        if use_polyak:
            ema = otu.tree_zeros_like(params)
        elif ema_bias_correction:
            ema = otu.tree_zeros_like(params)
        else:
            ema = params
        step = 0
        return (variance, ema, step)
    
    def update_fn(grads, state, params):
        variance, ema, step = state
        variance = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, variance, grads)
        step = step + 1
        variance_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance)
        updates = jax.tree_map(lambda m, v: m / (v ** 0.5 + eps), grads, variance_hat)
        if use_polyak:
            c = 1 / step
            ema = jax.tree_map(lambda e, p: c * p + (1 - c) * e, ema, params)
        else:
            ema = jax.tree_map(lambda e, p: ema_rate * e + (1 - ema_rate) * p, ema, params)
        return updates, (variance, ema, step)
    
    def get_ema_params(state):
        _, ema, step = state
        if use_polyak or not ema_bias_correction:
            return ema
        else:
            ema = jax.tree_map(lambda e: e / (1 - ema_rate ** step), ema)
            ema = jax.tree_map(lambda e: jnp.where(step == 0, 0, e), ema)
            return ema
    
    @jax.jit
    def get_grad_params(state, params):
        _, _, step = state
        ema_params = get_ema_params(state)
        return jax.tree_map(lambda p, e: p * (1-b1) + e * b1, params, ema_params)

    @jax.jit
    def get_eval_params(state, params):
        return get_ema_params(state)
    
    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, None)
