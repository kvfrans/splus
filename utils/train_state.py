###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# Contains model params and optimizer state.
class TrainStateEma(flax.struct.PyTreeNode):
    rng: Any
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    params_avg: Any
    tx: Any = nonpytree_field()
    opt_state: Any


    @classmethod
    def create(cls, model_def, params, rng, tx=None, opt_state=None, use_avg=False, **kwargs):
        if tx is not None and opt_state is None:
            opt_state = tx.init(params)
        params_avg = None if not use_avg else params

        return cls(
            rng=rng, step=1, apply_fn=model_def.apply, model_def=model_def, params=params, params_avg=params_avg,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    # Call model_def.apply_fn.
    def __call__(self, *args, params=None, method=None, **kwargs,):
        if params is None:
            params = self.params
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    def update_avg(self):
        c = 1 / self.step
        new_params_avg = jax.tree_map(lambda pa, p: c * p + (1 - c) * pa, self.params_avg, self.params)
        return self.replace(params_avg=new_params_avg)

    def call_model(self, *args, params=None, method=None, **kwargs):
        return self.__call__(*args, params=params, method=method, **kwargs)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'opt_state': self.opt_state,
            'step': self.step,
        }
    
    def load(self, data):
        return self.replace(**data)