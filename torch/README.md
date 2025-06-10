# SPlus Implementation in Pytorch

Here we present a single-file implementation of SPlus in Pytorch. See `splus.py`. To test the implementation, we provide a simple language model training script based on [nanoGPT](https://github.com/karpathy/nanoGPT).

You can run the script with:
```
python train.py --learning_rate=0.3 --wandb_run_name=splus-0.3 --optimizer_name=splus
```
and to compare against Adam:
```
python train.py --learning_rate=0.001 --wandb_run_name=adam-0.001 --optimizer_name=adam
```

<ins>This Pytorch implementation is for reference and is not optimized.</ins> Specifically, in the JAX implementation we support a distributed SPlus inversion step, but the Pytorch code does not do this.