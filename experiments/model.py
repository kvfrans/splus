import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import flax
from einops import rearrange
from dataclasses import field, dataclass
from flax.linen.dtypes import promote_dtype
import jax.lax as lax
Array, PRNGKey, Shape, Dtype = Any, Any, Tuple[int], Any
from helpers_model import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed, modulate, rms_norm, rms_normalize

global_dtype = jnp.bfloat16

# # Helper class that handles dtype and kernel initialization.
# class TransformerConfig():
#     normalize_activations: bool = False
#     normalize_weights: bool = False
#     normalize_embeddings: bool = False
#     use_scale_terms: bool = False

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)

#     def Dense(self, *args, **kwargs):
#         if self.normalize_weights:
#             return DenseWeightNorm(*args, **kwargs)
#             # return DenseWeightNorm(*args, **kwargs)
#         return nn.Dense(*args, **kwargs)

################################################################################
#                                 Input/Output Modules                         #
################################################################################

class TimestepEmbedder(nn.Module):
    """ Embeds scalar continuous time into vector representations. (For DiT)"""
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), use_bias=False, dtype=global_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), use_bias=False, dtype=global_dtype)(x)
        return x
    
    def timestep_embedding(self, t, max_period=10000):
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        embedding = embedding.astype(global_dtype) * jnp.sqrt(2) # RMS norm = 1.
        return embedding


class TokenEmbedder(nn.Module):
    """ Embed integer tokens into vector representations. """
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, labels):
        embedding_table = nn.Embed(self.num_classes, self.hidden_size, 
                            embedding_init=nn.initializers.normal(0.02), dtype=global_dtype)
        embeddings = embedding_table(labels)
        return embeddings
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    patch_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        print("INput to patch:", x.shape)
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.hidden_size, patch_tuple, patch_tuple, use_bias=False, padding="VALID", bias_init=nn.initializers.zeros_init(),
                     dtype=global_dtype)(x) # (B, P, P, hidden_size)
        print("Conv output:", x.shape)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        print("Output to patch:", x.shape)
        return x
    
class DiTOutput(nn.Module):
    """ The final layer of DiT. """
    patch_size: int
    channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.zeros_init(), 
                     bias_init=nn.initializers.zeros_init(), dtype=global_dtype)(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = nn.LayerNorm(use_bias=False, use_scale=False, dtype=global_dtype)(x)
        x = modulate(x, shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.channels, 
                     kernel_init=nn.initializers.zeros_init(), 
                     bias_init=nn.initializers.zeros_init(), dtype=global_dtype)(x)
        return x
    
class ClassifierOutput(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(use_bias=False, use_scale=True, dtype=global_dtype)(x)
        x = nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(0.02), use_bias=False, dtype=global_dtype)(x)
        return x

################################################################################
#                                 Transformer Blocks                           #
################################################################################

class Block(nn.Module):
    """ A block with adaptive layer norm zero (adaLN-Zero) conditioning. """
    hidden_size: int
    num_heads: int
    depth: int
    use_conditioning: bool
    use_causal_masking: bool
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        if self.use_conditioning:
            c = nn.silu(c) # Calculate adaLn modulation parameters.
            c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.normal(0.02), use_bias=False)(c)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        else:
            shift_msa, shift_mlp = (jnp.zeros_like(x[:, 0]) for _ in range(2))
            scale_msa, gate_msa, scale_mlp, gate_mlp = (jnp.ones_like(x[:, 0]) for _ in range(4))
        
        x_norm = nn.LayerNorm(use_bias=False, use_scale=not self.use_conditioning, dtype=global_dtype)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        channels_per_head = self.hidden_size // self.num_heads
        kqv = nn.Dense(3 * self.hidden_size, kernel_init=nn.initializers.normal(0.02), use_bias=False)(x_modulated)
        k, q, v = jnp.split(kqv, 3, axis=-1) 
        k = jnp.reshape(k, (k.shape[0], k.shape[1], self.num_heads, channels_per_head))
        q = jnp.reshape(q, (q.shape[0], q.shape[1], self.num_heads, channels_per_head))
        v = jnp.reshape(v, (v.shape[0], v.shape[1], self.num_heads, channels_per_head))
        q = q / jnp.sqrt(q.shape[3]) # 1/sqrt(d) scaling.
        w = jnp.einsum('bqhc,bkhc->bhqk', q, k) # [B, num_heads, Q, K]. Q,K = HW.
        w = w.astype(jnp.float32)
        if self.use_causal_masking:
            causal_mask = jnp.tri(N=w.shape[2], k=0) # [HW, HW].
            w = jnp.where(causal_mask[None, None, :, :], w, jnp.finfo(w.dtype).min)
            w = nn.softmax(w, axis=-1)
            w = jnp.where(causal_mask[None, None, :, :], w, 0)
        else:
            w = nn.softmax(w, axis=-1) # Softmax over key dimension = Total mass of 1 per query.
        info_max_attn_weight = jnp.mean(jnp.max(w, axis=-1))
        y = jnp.einsum('bhqk,bkhc->bqhc', w, v) # [B, Q=HW, num_heads, channels_per_head]
        y = jnp.reshape(y, x.shape) # [B, Q=HW, C] (C = heads * channels_per_head)
        x_attn = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02 / np.sqrt(2 * self.depth)), use_bias=False)(y)
        x_attn = gate_msa[:, None] * x_attn
        info_attn_norm_ratio = rms_norm(x_attn) / rms_norm(x)
        x = x + x_attn

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=not self.use_conditioning, dtype=global_dtype)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        x_mlp = nn.Dense(features=int(self.hidden_size * self.mlp_ratio), kernel_init=nn.initializers.normal(0.02), use_bias=False)(x_modulated2)
        info_relu_diff = jnp.mean(jnp.abs(jnp.mean((x_mlp > 0), axis=(0, 1)) - 0.5))
        info_relu_zero = jnp.mean(jnp.mean(x_mlp > 0, axis=(0, 1)) == 0)
        info_relu_positive = jnp.mean(jnp.mean(x_mlp > 0, axis=(0, 1)) == 1)
        info_relu_norm = rms_norm(x_mlp)
        x_mlp = nn.gelu(x_mlp)
        x_mlp = nn.Dense(features=x.shape[-1], kernel_init=nn.initializers.normal(0.02 / np.sqrt(2 * self.depth)), use_bias=False)(x_mlp)
        x_mlp = gate_mlp[:, None] * x_mlp
        info_mlp_norm_ratio = rms_norm(x_mlp) / rms_norm(x)
        x = x + x_mlp
        return x, (info_relu_diff, info_relu_zero, info_relu_positive, info_max_attn_weight, info_attn_norm_ratio, info_mlp_norm_ratio, info_relu_norm)


################################################################################
#                                 Main Model                                   #
################################################################################


class Transformer(nn.Module):
    """ Generic transformer, can become a DiT, ViT, or GPT. """
    train_type: str # 'dit', 'vit', 'gpt'
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_classes: int
    patch_size: int # DiT/ViT only

    @nn.compact
    def __call__(self, x, t, y, return_activations=False):
        print("Transformer: Input of shape", x.shape, "dtype", x.dtype)
        activations = {}
        infos = {}
        batch_size = x.shape[0]
        channels = x.shape[-1]

        if self.train_type == 'dit' or self.train_type == 'vit': # Patch embedding.
            input_size = x.shape[1]
            patch_side = input_size // self.patch_size
            num_patches = patch_side ** 2
            pos_embed = get_2d_sincos_pos_embed(None, self.hidden_size, num_patches)
            activations['pos_embed'] = pos_embed
            x = PatchEmbed(self.patch_size, self.hidden_size)(x)
            print("Transformer: After patch embed, shape is", x.shape, "dtype", x.dtype)
            activations['patch_embed'] = x
            x = x + pos_embed

        if self.train_type == 'dit': # Conditioning on timestep and label.
            ye = TokenEmbedder(self.num_classes + 1, self.hidden_size)(y) # +1 for unconditional class.
            te = TimestepEmbedder(self.hidden_size)(t)
            c = te + ye
            activations['time_embed'] = te
            activations['label_embed'] = ye
            activations['conditioning'] = c
        else:
            c = None

        if self.train_type == 'vit': # Add class token.
            class_token = TokenEmbedder(1, self.hidden_size)(jnp.zeros((batch_size, 1), dtype=jnp.int32))
            x = jnp.concatenate([class_token, x], axis=1)
            activations['class_token_embed'] = class_token

        if self.train_type == 'gpt': # Embed vocab.
            x = x.astype(jnp.uint32)
            x = TokenEmbedder(self.num_classes, self.hidden_size)(x)
            activations['vocab_embed'] = x
            pos_embed = TokenEmbedder(x.shape[1], self.hidden_size)(jnp.arange(x.shape[1])[None, :])
            activations['pos_embed'] = pos_embed
            x = x + pos_embed

        x = x.astype(global_dtype)
        seq_len = x.shape[1]
        activations[f'embed_input'] = x
        activations['normalized_input'] = x
        for i in range(self.depth):
            use_conditioning = self.train_type == 'dit'
            use_causal_masking = self.train_type == 'gpt'
            x, block_infos = Block(self.hidden_size, self.num_heads, self.depth, use_conditioning, use_causal_masking, self.mlp_ratio)(x, c)
            activations[f'block_{i}'] = x
            infos[f'block_{i}_relu_diff'] = block_infos[0]
            infos[f'block_{i}_relu_zero'] = block_infos[1]
            infos[f'block_{i}_relu_positive'] = block_infos[2]
            infos[f'block_{i}_max_attn_weight'] = block_infos[3]
            infos[f'block_{i}_attn_norm_ratio'] = block_infos[4]
            infos[f'block_{i}_mlp_norm_ratio'] = block_infos[5]
            infos[f'block_{i}_relu_norm'] = block_infos[6]

        if self.train_type == 'dit':
            x = DiTOutput(self.patch_size, channels, self.hidden_size)(x, c) # (B, num_patches, p*p*c)
            x = jnp.reshape(x, (batch_size, patch_side, patch_side, self.patch_size, self.patch_size, channels))
            x = jnp.einsum('bhwpqc->bhpwqc', x)
            x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=patch_side, W=patch_side)
            assert x.shape == (batch_size, input_size, input_size, channels)
        elif self.train_type == 'vit':
            class_token_vec = x[:, 0]
            x = ClassifierOutput(self.num_classes)(class_token_vec)
            assert x.shape == (batch_size, self.num_classes)
        elif self.train_type == 'gpt':
            x = ClassifierOutput(self.num_classes)(x)

        activations['final_layer'] = x
        if return_activations:
            return x, activations, infos
        return x
