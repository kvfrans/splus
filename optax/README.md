# SPlus Implementation in Optax

Here we present a single-file implementation of SPlus using the [Optax](https://optax.readthedocs.io/en/latest/index.html) library for JAX training. See `splus.py`. To test the implementation, we provide a simple language model training script based on [jaxtransformer](https://github.com/kvfrans/jaxtransformer).

You can run the script with:
```
python test_splus.py --train.dataset_name openwebtext --wandb.project splus_optax --wandb.name GPT-SPlus --tf.hidden_size 128 --tf.depth 6 --tf.num_heads 4 --tf.mlp_ratio 2 --train.batch_size 8
```
and to compare against Adam:
```
python test_splus.py --train.dataset_name openwebtext --wandb.project splus_optax --wandb.name GPT-Adam --tf.hidden_size 128 --tf.depth 6 --tf.num_heads 4 --tf.mlp_ratio 2 --optimizer adam --lr 0.001 --train.batch_size 8
```

Note, *this is not the training script we use in the paper.* This folder is just to provide an example of how to use SPlus with Optax. For the exact experimental script, see `experiments/` from the base of the repo. 