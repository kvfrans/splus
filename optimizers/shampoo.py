import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
import jax.numpy as jnp
from optimizers.optimizer_utils import Optimizer

def make_shampoo(
        b1: float = 0.9,
        b2: float = 0.999,
        max_dim: int = 10000,
        eps: float = 1e-8,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        variance = otu.tree_zeros_like(params)
        def sides_decomp(p):
            if len(p.shape) == 2:
                if p.shape[0] > max_dim or p.shape[1] > max_dim:
                    return None
                return [jnp.zeros((d, d)) for d in p.shape]
            return None
        sides = jax.tree_map(sides_decomp, params)
        def qs_decomp(p):
            if len(p.shape) == 2:
                if p.shape[0] > max_dim or p.shape[1] > max_dim:
                    return None
                return [jnp.eye(d) for d in p.shape]
            return None
        q_sides = jax.tree_map(qs_decomp, params)
        step = 0
        return (momentum, variance, sides, q_sides, step)
    
    def get_grad_params(state, params):
        return params
    
    def get_eval_params(state, params):
        return params

    def update_sides(g, s):
        if len(g.shape) == 2 and s is not None:
            ggT = (g @ g.T) / g.shape[1]
            gTg = (g.T @ g) / g.shape[0]
            return [
                b2 * s[0] + (1 - b2) * ggT if s[0] is not None else None,
                b2 * s[1] + (1 - b2) * gTg if s[1] is not None else None,
            ]
        else:
            return None
        
    def precondition(p, q, s, v):
        if len(p.shape) == 2 and q is not None:
            p = q[0] @ p if q[0] is not None else p
            p = p @ q[1] if q[1] is not None else p
            p *= jnp.mean(jnp.diag(jnp.sqrt(s[0])))
            return p
        else:
            return p / (jnp.sqrt(v) + eps)

    def update_fn(grads, state, params):
        momentum, variance, sides, q_sides, step = state
        step = step + 1

        variance = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, variance, grads)
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        variance_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance)
        momentum_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum)
        momentum_hat = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), momentum_hat, variance)
        grads = jax.tree_map(lambda g, v: g / (jnp.sqrt(v) + eps), grads, variance)
        updates = jax.tree_map(precondition, momentum_hat, q_sides, sides, variance_hat)            
        sides = jax.tree_map(update_sides, grads, sides)

        updates = jax.tree_map(lambda u: jnp.where(step < 30, u*0, u), updates)

        print("Finished update_fn.")
        return updates, (momentum, variance, sides, q_sides, step)
    
    @jax.jit # We JIT the individual QR updates since it compiles faster than JITting the entire tree_map.
    def do_inverse(s):
        print("Matrix inverse for", s.shape)
        if s is None:
            return None 
        eigvals, eigvecs = jnp.linalg.eigh(s + 1e-30 * jnp.eye(s.shape[0]))
        eigvals = jnp.abs(eigvals)
        eigvals = 1 / (jnp.sqrt(eigvals) + 1e-8)
        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
            
    def update_slow(state, sharding, **kwargs):
        _, _, sides, q_sides, step = state
        devices = jax.local_devices()
        tensor_shapes = {}
        def put_device_staggered(p):
            idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
            tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
            return jax.device_put(p, devices[idx])
        sides = jax.experimental.multihost_utils.process_allgather(sides)
        sides = jax.tree_map(put_device_staggered, sides)
        sides = jax.tree_map(lambda s: s / (1 - b2 ** step), sides)
        
        q_sides = jax.tree_map(do_inverse, sides)
        q_sides = jax.tree_map(lambda x: jax.device_get(x), q_sides)
        q_sides = jax.jit(lambda x: x, out_shardings=sharding[3])(q_sides)
        return (state[0], state[1], state[2], q_sides, state[4])

    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, update_slow, None)