from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Callable, Optional, Union
import chex
from optax import tree_utils as otu
from optax._src import base, combine, transform

class SPlusState(NamedTuple):
    ema: chex.Array
    momentum: chex.Array
    sides: chex.Array
    q_sides: chex.Array
    step: int
    ema_rate: float

def splus_get_eval_params(state):
    ema_hat = jax.tree_map(lambda e: e / (1 - state.ema_rate ** state.step), state.ema)
    return ema_hat

def splus(
        learning_rate: base.ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        ema_rate: float = 0.999,
        eps: float = 1e-30,
        inverse_every: int = 100,
        nonstandard_constant: float = 0.001,
        nonstandard_strings: list[str] = ['embed', 'layernorm'],
        weight_decay: float = 1e-2,
        mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
        jit_broadcast_computation: bool = False,
        jit_original_sharding: SPlusState | None = None,
        max_dim: int = 10000,
        verbose: bool = True,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        ema = otu.tree_zeros_like(params)
        def sides_decomp(p):
            if len(p.shape) == 2:
                return [jnp.zeros((d, d)) if d < max_dim 
                        else None for d in p.shape]
            return None
        sides = jax.tree_map(sides_decomp, params)
        def qs_decomp(p):
            if len(p.shape) == 2:
                return [jnp.eye(d) if d < max_dim 
                        else None for d in p.shape]
        q_sides = jax.tree_map(qs_decomp, params)
        step = 0
        return SPlusState(ema, momentum, sides, q_sides, step, ema_rate)

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
    def get_eigvecs(s):
        if s is None:
            return None
        if verbose:
            print("(SPlus) Compiling eigevec decomposition for shape", s.shape)
        _, q = jnp.linalg.eigh(s + eps * jnp.eye(s.shape[0]))
        return q
    
    def update_inverse(sides):
        if jit_broadcast_computation:
            devices = jax.local_devices()
            tensor_shapes = {}
            def put_device_staggered(p):
                idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
                tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
                return jax.device_put(p, devices[idx])
            sides = jax.experimental.multihost_utils.process_allgather(sides)
            sides = jax.tree_map(put_device_staggered, sides)
        q_sides = jax.tree_map(get_eigvecs, sides)
        if jit_broadcast_computation:
            q_sides = jax.tree_map(lambda _, x: jax.device_get(x), sides, q_sides)
            if jit_original_sharding is not None:
                q_sides = jax.jit(lambda x: x, out_shardings=jit_original_sharding.q_sides)(q_sides)
        return q_sides
    
    def update_fn(grads, state, params):
        step = state.step + 1

        # Rotate to eigenbasis, take sign, unrotate.
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.momentum, grads)
        momentum_rot = jax.tree_map(rot, momentum, state.q_sides)
        # momentum_rot_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum_rot)
        updates_rot = jax.tree_map(lambda m: jnp.sign(m), momentum_rot)
        updates = jax.tree_map(unrot, updates_rot, state.q_sides)
        sides = jax.tree_map(update_sides, grads, state.sides)
        ema = jax.tree_map(lambda e, g: ema_rate * e + (1 - ema_rate) * g, state.ema, params)

        # Every `inverse_every` steps, we update the inverse eigendecomposition.
        do_inverse = (step % inverse_every == 0) | (step == 1)
        q_sides = jax.lax.cond(do_inverse, update_inverse, lambda _ : state.q_sides, sides)

        return updates, SPlusState(ema, momentum, sides, q_sides, step, state.ema_rate)
    
    def shape_scaling(updates, state, params):
        def shape_scale(path, u):
            path_str = '/'.join([p.key for p in path])
            if len(u.shape) == 2 and not any([k in path_str.lower() for k in nonstandard_strings]) and u.shape[0] < max_dim and u.shape[1] < max_dim:
                scale = (1 / (u.shape[0] + u.shape[1])/2)
            else:
                scale = nonstandard_constant
            return u * scale
        return jax.tree_util.tree_map_with_path(shape_scale, updates), None
    
    splus_main = base.GradientTransformation(init_fn, update_fn)
    splus_scaling = base.GradientTransformation(lambda _ : None, shape_scaling)
    return combine.chain(
        splus_main,
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
        splus_scaling
    )