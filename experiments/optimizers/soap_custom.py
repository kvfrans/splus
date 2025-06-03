import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
import jax.numpy as jnp
from optimizers.optimizer_utils import Optimizer

def make_soap_custom(
        b1: float = 0.9,
        b2: float = 0.999,
        ema_rate: float = 0.999,
        use_ema: bool = False,
        ema_interp: float = 0.0,
        clip_elementwise_update: float = 100.0,
        max_dim: int = 10000,
        eps: float = 1e-8,
        normalization_type: str = 'rms',
        soap_side: str = 'all', # or ['start', 'end', 'none']
        clip: float = 0.05,
        batch_size: int = 128, # for sophia scaling.
        qr_style: str = 'ggt', # or 'covariance'
        shampoo_diagonal: str = 'none',
        factor_ratio: float = 1.0,
        factor_lr: float = 1.0,
    ):

    def should_precondition(shape):
        assert len(shape) == 2
        if soap_side == 'all':
            return [d < max_dim for d in shape]
        elif soap_side == 'start':
            return [shape[0] < max_dim, False]
        elif soap_side == 'end':
            return [False, shape[1] < max_dim]
        elif soap_side == 'none':
            return [False, False]
        raise ValueError(f'SOAP side {soap_side} not recognized')

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        ema = otu.tree_zeros_like(params)
        variance_rot = otu.tree_zeros_like(params)
        variance_pre = otu.tree_zeros_like(params)
        def sides_decomp(p):
            if len(p.shape) == 2:
                return [jnp.zeros((d, d)) if should_precondition(p.shape)[i] else None for i, d in enumerate(p.shape)]
            return None
        sides = jax.tree_map(sides_decomp, params)
        def qs_decomp(p):
            if len(p.shape) == 2:
                return [jnp.eye(d) if should_precondition(p.shape)[i] else None for i, d in enumerate(p.shape)]
            return None
        q_sides = jax.tree_map(qs_decomp, params)
        step = 0
        return (ema, momentum, variance_rot, variance_pre, sides, q_sides, step)

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
        if ema_interp != 0:
            return jax.tree_map(lambda p, e: (1 - ema_interp) * p + ema_interp * e, params, state[0])
        return params
    
    @jax.jit
    def get_eval_params(state, params):
        ema = state[0]
        step = state[6]
        if use_ema:
            ema_hat = jax.tree_map(lambda e: e / (1 - ema_rate ** step), ema)
            return ema_hat
        else:
            return params
    
    def get_update(momentum, variance):
        if normalization_type == 'rms':
            return jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), momentum, variance)
        elif normalization_type == 'divide':
            return jax.tree_map(lambda m, v: jnp.clip(m / (v + eps), 0, 100), momentum, variance)
        elif normalization_type == 'divideclip':
            return jax.tree_map(lambda m, v: jnp.clip(m / jnp.maximum(clip * v * batch_size, eps), -1, 1), momentum, variance)
        elif normalization_type == 'sign':
            return jax.tree_map(lambda m: jnp.sign(m), momentum)
        elif normalization_type == '0.5':
            return jax.tree_map(lambda m: jnp.abs(m)**0.5 * jnp.sign(m), momentum)
        elif normalization_type == '0.25':
            return jax.tree_map(lambda m: jnp.abs(m)**0.25 * jnp.sign(m), momentum)
        elif normalization_type == '0.75':
            return jax.tree_map(lambda m: jnp.abs(m)**0.75 * jnp.sign(m), momentum)
        elif normalization_type == 'none':
            return momentum
        elif normalization_type == 'fixednorm':
            return jax.tree_map(lambda m: m / jnp.sqrt(jnp.mean(jnp.square(m))), momentum)
        elif normalization_type == 'fixednormclip':
            return jax.tree_map(lambda m: jnp.clip(m / jnp.sqrt(jnp.mean(jnp.square(m))), -10, 10), momentum)
        elif normalization_type == 'fixednormclipone':
            return jax.tree_map(lambda m: jnp.clip(m / jnp.sqrt(jnp.mean(jnp.square(m))), -1, 1), momentum)
        elif normalization_type == 'fixednormsqrt':
            return jax.tree_map(lambda m: jnp.sqrt(m) / jnp.sqrt(jnp.mean(m)), momentum)
        raise ValueError(f'Normalization type {normalization_type} not recognized')
    
    def update_fn(grads, state, params):
        ema, momentum, variance_rot, variance_pre, sides, q_sides, step = state

        step = step + 1
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        variance_pre = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g ** 2, variance_pre, grads)
        variance_pre_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance_pre)

        if shampoo_diagonal == 'sign':
            grads = jax.tree_map(lambda g: jnp.sign(g), grads)
            momentum = jax.tree_map(lambda m: jnp.sign(m), momentum)
        elif shampoo_diagonal == 'rms':
            grads = jax.tree_map(lambda g, v: g / (jnp.sqrt(v) + eps), grads, variance_pre_hat)
            momentum = jax.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), momentum, variance_pre_hat)

        grad_rot = jax.tree_map(rot, grads, q_sides)
        momentum_rot = jax.tree_map(rot, momentum, q_sides)
        variance_rot = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g ** 2, variance_rot, grad_rot)

        momentum_rot_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum_rot)
        variance_rot_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance_rot)

        updates_rot = get_update(momentum_rot_hat, variance_rot_hat)
        updates = jax.tree_map(unrot, updates_rot, q_sides)
        sides = jax.tree_map(update_sides, grads, sides)
        updates = jax.tree_map(lambda u: jnp.where(step == 1, 0, u), updates)
        momentum = jax.tree_map(lambda m: jnp.where(step == 1, 0, m), momentum)
        variance_rot = jax.tree_map(lambda v: jnp.where(step == 1, 0, v), variance_rot)
        ema = jax.tree_map(lambda e, g: ema_rate * e + (1 - ema_rate) * g, ema, params)

        updates = jax.tree_map(lambda u: jnp.clip(u, -clip_elementwise_update, clip_elementwise_update), updates)

        print("Finished update_fn.")
        return updates, (ema, momentum, variance_rot, variance_pre, sides, q_sides, step)
    
    @jax.jit # We JIT the individual QR updates since it compiles faster than JITting the entire tree_map.
    def do_qr_update(s):
        print("QR update for shape", s.shape)
        if s is None:
            return None 
        eigvals, q = jnp.linalg.eigh(s + 1e-30 * jnp.eye(s.shape[0]))
        return q, eigvals
    
    def qr_factor_update(s):
        if factor_ratio == 1.0:
            return do_qr_update(s)
        else:
            q_full = jnp.eye(s.shape[0]) * jnp.sqrt(factor_lr)
            num_factors = int(jnp.ceil(s.shape[0] * factor_ratio))
            if num_factors > 0:
                s_idx = jnp.argsort(jnp.diag(s))[::-1]
                s_permuted = s[s_idx][:, s_idx]
                s_factors = s_permuted[:num_factors, :num_factors]
                q_factors, _ = do_qr_update(s_factors)
                q_factors = jax.tree_map(lambda q: jax.device_get(q), q_factors)
                q_full = q_full.at[s_idx[:num_factors], :num_factors].set(q_factors)
            return q_full, None
    
    def update_slow(state, sharding, **kwargs):
        _, momentum, _, _,  sides, q_sides, _ = state
        devices = jax.local_devices()
        tensor_shapes = {}
        def put_device_staggered(p):
            idx = tensor_shapes.get(p.shape, 0) % jax.local_device_count()
            tensor_shapes[p.shape] = tensor_shapes.get(p.shape, 0) + 1
            return jax.device_put(p, devices[idx])

        if qr_style == 'covariance':
            @jax.jit
            def subtract_means(m, s):
                if len(m.shape) == 2:
                    return [
                        (s[0] - m @ m.T) if s[0] is not None else None,
                        (s[1] - m.T @ m) if s[1] is not None else None,
                    ]
                else:
                    return None
            sides = jax.tree_map(subtract_means, momentum, sides)

        sides = jax.experimental.multihost_utils.process_allgather(sides)
        sides = jax.tree_map(put_device_staggered, sides)
        q_sides_eigvals = jax.tree_map(qr_factor_update, sides)
        q_sides = jax.tree_map(lambda _, x: jax.device_get(x[0]), sides, q_sides_eigvals)
        eigvals = jax.tree_map(lambda _, x: jax.device_get(x[1]), sides, q_sides_eigvals)

        q_sides = jax.jit(lambda x: x, out_shardings=sharding[3])(q_sides)


        return (state[0], state[1], state[2], state[3], state[4], q_sides, state[6])

    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, update_slow, None)