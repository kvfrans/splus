import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
import jax.numpy as jnp
from optimizers.optimizer_utils import Optimizer

def make_soap(
        b1: float = 0.9,
        b2: float = 0.999,
        max_dim: int = 10000,
        eps: float = 1e-8,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        variance_rot = otu.tree_zeros_like(params)
        def sides_decomp(p):
            if len(p.shape) == 2:
                return [jnp.zeros((d, d)) if d < max_dim else None for d in p.shape]
            return None
        sides = jax.tree_map(sides_decomp, params)
        def qs_decomp(p):
            if len(p.shape) == 2:
                return [jnp.eye(d) if d < max_dim else None for d in p.shape]
            return None
        q_sides = jax.tree_map(qs_decomp, params)
        step = 0
        return (momentum, variance_rot, sides, q_sides, step)

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
    
    def get_grad_params(state, params):
        return params
    
    def get_eval_params(state, params):
        return params
    
    def update_fn(grads, state, params):
        momentum, variance_rot, sides, q_sides, step = state

        step = step + 1
        
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        grad_rot = jax.tree_map(rot, grads, q_sides)
        momentum_rot = jax.tree_map(rot, momentum, q_sides)
        variance_rot = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g ** 2, variance_rot, grad_rot)

        momentum_rot_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum_rot)
        variance_rot_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance_rot)
        updates_rot = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), momentum_rot_hat, variance_rot_hat)
        updates = jax.tree_map(unrot, updates_rot, q_sides)
        sides = jax.tree_map(update_sides, grads, sides)
        updates = jax.tree_map(lambda u: jnp.where(step == 1, 0, u), updates)
        momentum = jax.tree_map(lambda m: jnp.where(step == 1, 0, m), momentum)
        variance_rot = jax.tree_map(lambda v: jnp.where(step == 1, 0, v), variance_rot)

        print("Finished update_fn.")
        return updates, (momentum, variance_rot, sides, q_sides, step)
    
    @jax.jit # We JIT the individual QR updates since it compiles faster than JITting the entire tree_map.
    def do_qr_update(s):
        print("QR update for shape", s.shape)
        if s is None:
            return None 
        q = jnp.linalg.eigh(s + 1e-30 * jnp.eye(s.shape[0]))[1]
        return q
    
    def update_slow(state, sharding, **kwargs):
        _, _, sides, q_sides, _ = state
        devices = jax.local_devices()
        tensor_shapes = {}
        def put_device_staggered(p):
            idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
            tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
            return jax.device_put(p, devices[idx])
        sides = jax.experimental.multihost_utils.process_allgather(sides)
        sides = jax.tree_map(put_device_staggered, sides)
        q_sides = jax.tree_map(do_qr_update, sides)
        q_sides = jax.tree_map(lambda x: jax.device_get(x), q_sides)
        q_sides = jax.jit(lambda x: x, out_shardings=sharding[3])(q_sides)
        return (state[0], state[1], state[2], q_sides, state[4])

    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, update_slow, None)