import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
from optax._src import base as optax_base
import jax
import jax.numpy as jnp
from optimizers.optimizer_utils import Optimizer
from typing import NamedTuple, Any, Callable, Optional, Union

def make_splus(
        b1: float = 0.9,
        b2: float = 0.999,
        ema_rate: float = 0.999,
        eps: float = 1e-30,
        max_dim: int = 10000,
    ):

    def should_precondition(shape):
        return [d < max_dim for d in shape]

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        ema = otu.tree_zeros_like(params)
        def sides_decomp(p):
            if len(p.shape) == 2:
                return [jnp.zeros((d, d)) if should_precondition(p.shape)[i] 
                        else None for i, d in enumerate(p.shape)]
            return None
        sides = jax.tree_map(sides_decomp, params)
        def qs_decomp(p):
            if len(p.shape) == 2:
                return [jnp.eye(d) if should_precondition(p.shape)[i] 
                        else None for i, d in enumerate(p.shape)]
            return None
        q_sides = jax.tree_map(qs_decomp, params)
        step = 0
        return ema, momentum, sides, q_sides, step

    def update_sides(g, s):
        if len(g.shape) == 2:
            return [
                b2 * s[0] + (1 - b2) * g @ g.T if s[0] is not None else None,
                b2 * s[1] + (1 - b2) * g.T @ g if s[1] is not None else None,
            ]
        else:
            return None

    def rot(p, q):
        if len(p.shape) == 2:
            p = q[0].T @ p if q[0] is not None else p
            p = p @ q[1] if q[1] is not None else p
        return p
    
    def unrot(p, q):
        if len(p.shape) == 2:
            p = q[0] @ p if q[0] is not None else p
            p = p @ q[1].T if q[1] is not None else p
        return p
    
    @jax.jit
    def get_grad_params(state, params):
        return params
    
    @jax.jit
    def get_eval_params(state, params):
        ema, _, _, _, step = state
        ema_hat = jax.tree_map(lambda e: e / (1 - ema_rate ** step), ema)
        return ema_hat
    
    def update_fn(grads, state, params):
        ema, momentum, sides, q_sides, step = state

        step = step + 1
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        momentum_rot = jax.tree_map(rot, momentum, q_sides)
        momentum_rot_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum_rot)
        updates_rot = jax.tree_map(lambda m: jnp.sign(m), momentum_rot_hat)
        updates = jax.tree_map(unrot, updates_rot, q_sides)
        sides = jax.tree_map(update_sides, grads, sides)
        ema = jax.tree_map(lambda e, g: ema_rate * e + (1 - ema_rate) * g, ema, params)
        return updates, (ema, momentum, sides, q_sides, step)
    
    @jax.jit # We JIT the individual QR updates since it compiles faster than JITting the entire tree_map.
    def do_qr_update(s):
        if s is None:
            return None 
        _, q = jnp.linalg.eigh(s + eps * jnp.eye(s.shape[0]))
        return q
    
    def update_slow(state, sharding, **kwargs):
        _, _, sides, _, _ = state

        devices = jax.local_devices()
        tensor_shapes = {}
        def put_device_staggered(p):
            idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
            tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
            return jax.device_put(p, devices[idx])

        sides = jax.experimental.multihost_utils.process_allgather(sides)
        sides = jax.tree_map(put_device_staggered, sides)
        q_sides = jax.tree_map(do_qr_update, sides)
        q_sides = jax.tree_map(lambda _, x: jax.device_get(x), sides, q_sides)
        q_sides = jax.jit(lambda x: x, out_shardings=sharding[3])(q_sides)

        return state[0], state[1], state[2], q_sides, state[4]

    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, update_slow, None)