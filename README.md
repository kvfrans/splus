# SPlus: A Stable Whitening Optimizer for Neural Network Optimization

This work introduces **SPlus**, a new optimizer for neural network training. We developed SPlus from an fundamentally empirical and experimental point of view -- the goal was to make a practical optimizer that just works, and is consistently fast in terms of both gradient steps and wallclock speed. SPlus is based off the Shampoo family of algorithms, and its main speedup comes from performing gradient descent over a "whitening" distance metric. In addition, we introduce some critical stabilization techniques.

In our experiments, SPlus matches the performance of Adam with **44%** of the gradient steps, and **62%** of the wall-clock time. We tested on language modelling, diffusion modelling, and image classification objective; all using a standard Transformer architecture. Please give SPlus a try on your problem setting.

For more details on the algorithm, read the paper: [ArXiv Link Todo].

## How do I use SPlus in my existing training setup?
We provide single-file implementations of SPlus in both JAX and Torch. See `optax/splus.py` and `torch/splus.py`. We designed things to be easily plug-and-play, but <ins>please follow the following instructions. You will need to add a line to evaluate with the EMA parameters, and you will need to adjust your LR.<ins>

### JAX instructions

Put the `splus.py` file in your project directory. Then you can simply replace `optax.adamw` with `splus`:
```{python}
# Replace the optax Adam:
import optax
tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001, mask=weight_decay_mask)

# With SPlus:
from splus import splus, splus_get_eval_params
tx = splus(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=0.001, mask=weight_decay_mask)
```
> [!IMPORTANT] 
> SPlus uses a **different set of evaluation parameters** than during training. To support this, it is important to use the helper function `splus_get_eval_params`:
```
splus_state = train_state.opt_state[0]
train_state_eval = train_state.replace(params=splus_get_eval_params(splus_state))
get_validation_loss(train_state_eval)
```
Change your LR as described in the LR section. See `optax/train.py` for an example. 

## Pytorch instructions

Put the `splus.py` file in your project directory. Then you can simply replace `optax.adamw` with `splus`:
```{python}
# Replace the torch Adam:
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# With SPlus:
from splus import SPlus
optimizer = SPlus(optim_groups, lr=learning_rate, b1=beta1, b2=beta2, weight_decay=weight_decay)
```
> [!IMPORTANT] 
> Important: SPlus uses a **different set of evaluation parameters** than during training. To support this, use the helper functions `optimizer.eval()` and `optimizer.train()`
```
# Training step
optimizer.train() # New in SPlus: Include this!
model.train()
loss = model(batch)
loss.backwards()
optimizer.step()
optimizer.zero_grad()

# Eval step
optimizer.eval() # New in SPlus: Include this!
validation_loss = model(validation_batch)
```
Change your LR as described in the LR section. See `torch/train.py` for an example.

## How do I choose the learning rate?
> [!IMPORTANT] 
> SPlus uses a different learning rate scale than Adam, so you need to change your learning rate.
If you already have a tuned Adam implementation, then use the following formula for a rough LR:
```
splus_lr = adam_lr * (network hidden size) * 2
```
Alternatively, for training common transformer models, you can try a simple:
```
splus_lr = 0.2
```
Of course, please do a proper initial sweep as the optimal LR depends on batch size, data complexity, etc.
SPlus learning rates are consistent across network width. When you scale up your network size, you can use the same LR.


## How do I reproduce the experiments in the paper?
The full code to replicate the paper is in the `experiments/` directory.  In the experiments, we train a standard Transformer architecture on three objectives -- language modelling, diffusion modelling, and image classification.
The experiment code is designed to be run on TPUs. We ran all experiments on a pod of 32 TPU-v3 machines.
To fairly compare against other methods, we've re-implemented many popular optimizers in JAX. See `experiments/optimizers` for:
- Adam
- Muon
- PSGD
- SOAP
- Shampoo
- Sophia
- Schedule-Free Adam
We tried our best to faithfully capture implementation details and hyperparameters. 

Feel free to use our experimental code for further research on neural network optimizers. For researchers, some pointers:
- Make sure to compare performance starting from various trained checkpoints, not just from initialization.
- Use a large enough batch size. We use `1024` in the paper.
- Always sweep over learning rates independently for every method. Yes, this can take a while, but it is neccessary to accurately compare methods.
