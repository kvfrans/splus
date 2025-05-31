import jax.experimental
import jax.experimental.multihost_utils
from optax import tree_utils as otu
import jax
from optimizers.optimizer_utils import Optimizer
import jax.numpy as jnp

def make_sophia(
        b1: float = 0.9,
        b2: float = 0.999,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):

    def init_fn(params):
        momentum = otu.tree_zeros_like(params)
        variance = otu.tree_zeros_like(params)
        step = 0
        return (momentum, variance, step)
    
    def update_fn(grads, state, params):
        del params
        momentum, variance, step = state
        momentum = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, momentum, grads)
        step = step + 1
        momentum_hat = jax.tree_map(lambda m: m / (1 - b1 ** step), momentum)
        variance_hat = jax.tree_map(lambda v: v / (1 - b2 ** step), variance)
        updates = jax.tree_map(lambda m, v: jnp.clip(m / jnp.maximum(clip * v, eps), -1, 1), momentum_hat, variance_hat)
        return updates, (momentum, variance, step)
    
    @jax.jit
    def do_update(state, train_state, batch):
        _, variance, _ = state
        text_input = batch[0][:, :-1]
        def loss_fn_sophia(params):
            logits = train_state.call_model(text_input, None, None, params=params)
            log_probs = jax.nn.log_softmax(logits)
            sampled_targets = jax.random.categorical(train_state.rng, logits)
            loss = jnp.mean(jnp.sum(-log_probs * jax.nn.one_hot(sampled_targets, logits.shape[-1]), axis=-1))
            return loss
        grads = jax.grad(loss_fn_sophia)(train_state.params)
        variance = jax.tree_map(lambda v, g: b2 * v + (1 - b2) * g**2, variance, grads)
        return (state[0], variance, state[2])

    def update_slow(state, sharding, train_state, batch):
        return do_update(state, train_state, batch)
    
    def get_grad_params(state, params):
        return params

    def get_eval_params(state, params):
        return params

    return Optimizer(init_fn, update_fn, get_grad_params, get_eval_params, update_slow, None)