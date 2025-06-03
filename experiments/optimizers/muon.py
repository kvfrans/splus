from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optimizers.optimizer_utils import Optimizer


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    v: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2 or (x.shape[0] > 10000) or (x.shape[1] > 10000):
    return x / (v ** 0.5 + eps)
  # if x.shape[0] > 10000 or x.shape[1] > 10000:
    # return x * 1
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )
  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x
  transposed = False
  if x.shape[0] > x.shape[1]:
    x = x.T
    transposed = True
  x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
  ns_coeffs = ns_coeffs.astype(x.dtype)
  if ns_coeffs.ndim == 1:
    x = jax.lax.fori_loop(
        0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
    )
  else:
    x, _ = jax.lax.scan(
        lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
    )
  if transposed:
    x = x.T
  x = jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x
  return x


class MuonState(NamedTuple):
  """State for the Muon algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  v: base.Updates
  ns_coeffs: chex.Array  # shape=(), dtype=jnp.int32.


def make_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    adaptive: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    v = otu.tree_zeros_like(params)  # Second moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        v=v,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    mu = jax.tree_map(lambda m, g: beta * m + (1 - beta) * g, state.mu, updates)
    v = jax.tree_map(
        lambda m, g: beta2 * m + (1 - beta2) * g**2, state.v, updates
    )
    count = state.count + 1
    mu_hat = jax.tree_map(lambda m: m / (1 - beta ** count), mu)
    v_hat = jax.tree_map(lambda m: m / (1 - beta2 ** count), v)
    updates = jax.tree_map(
        lambda x, v: orthogonalize_via_newton_schulz(
            x, v, state.ns_coeffs, ns_steps, eps
        ),
        mu_hat, v_hat,
    )
    return updates, MuonState(
        count=count,
        mu=mu,
        v=v,
        ns_coeffs=state.ns_coeffs,
    )
  
  def get_grad_params(state, params):
    return params

  def get_eval_params(state, params):
    return params

  return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, None, None)
