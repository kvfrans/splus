import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

def rms_norm(x, axis=None, keepdims=False):
    return jnp.sqrt(jnp.mean(x**2, axis=axis, keepdims=keepdims) + 1e-6)

def rms_normalize(x, axis=-1):
    norm = rms_norm(x, axis=axis, keepdims=True) + 1e-4
    return x / norm

def xavier_uniform_pytorchlike():
    def init(key, shape, dtype):
        from jax._src import core
        from jax._src import dtypes
        dtype = dtypes.canonicalize_dtype(dtype)
        if len(shape) == 2: # Dense, [in, out]
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # Conv, [k, k, in, out]. Assumes patch-embed style conv.
            fan_in = shape[0] * shape[1] * shape[2]
            fan_out = shape[3]
        else:
            raise ValueError(f"Invalid shape {shape}")

        variance = 2 / (fan_in + fan_out)
        scale = jnp.sqrt(3 * variance)
        param = jax.random.uniform(key, shape, dtype, -1) * scale

        return param
    return init

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32))
    return jnp.expand_dims(emb,0)

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0) # (1, H*W, D)

def apply_rope(q, k, pos, theta=10000):
    dim = q.shape[-1]
    scale = jnp.arange(0, dim, 2) / dim
    omega = 1.0 / (theta**scale)
    freqs = jnp.einsum("...n,d->...nd", pos, omega)
    freqs = jnp.stack([jnp.cos(freqs), -jnp.sin(freqs), jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
    freqs = rearrange(freqs, "b n d (i j) -> b n d i j", i=2, j=2)
    qr = jnp.reshape(q, (*q.shape[:-1], -1, 1, 2))
    kr = jnp.reshape(k, (*k.shape[:-1], -1, 1, 2))
    q_out = freqs[..., 0] * qr[..., 0] + freqs[..., 1] * qr[..., 1]
    k_out = freqs[..., 0] * kr[..., 0] + freqs[..., 1] * kr[..., 1]
    return jnp.reshape(q_out, q.shape), jnp.reshape(k_out, k.shape)