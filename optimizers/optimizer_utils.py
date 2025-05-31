from typing import Callable, NamedTuple

class Optimizer(NamedTuple):
    init: Callable
    update: Callable
    get_grad_params: Callable
    get_eval_params: Callable
    update_slow: Callable = None
    log_stats: Callable = None