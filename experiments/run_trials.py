job_list = []
debug_config = '--model.hidden_size 64 --model.depth 2 --model.num_heads 2 --model.mlp_ratio 1'
tiny_config = '--model.hidden_size 128 --model.depth 4 --model.num_heads 4 --model.mlp_ratio 4' 
xsmall_config = '--model.hidden_size 144 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
small_config = '--model.hidden_size 384 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
big_config = '--model.hidden_size 768 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
large_config = '--model.hidden_size 1024 --model.depth 24 --model.num_heads 16 --model.mlp_ratio 4'
xlarge_config = '--model.hidden_size 1152 --model.depth 28 --model.num_heads 16 --model.mlp_ratio 4'

dit_config = '--model.train_type dit --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1 --model.patch_size 2 --model.use_stable_vae 1'
vit_config = '--model.train_type vit --dataset_name imagenet256-augment --model.patch_size 16 --model.use_stable_vae 0'
gpt_config = '--model.train_type gpt --dataset_name openwebtext --model.use_stable_vae 0'

###
### This file is for reference on how to compare the various optimizers. You'll need to setup your own system for scheduling the commands. 
### You can use our reference checkpoints at this gcp bucket: gs://rll-kvfrans-public/splus_checkpoints
###

base = f'python train.py --wandb.group SPlusExperiments --model.sharding fsdp --log_interval 10 --max_steps 10_000 --model.warmup 200 --model.weight_decay 0.1 {big_config} --model.sequence_length 256 --batch_size 1024'
for ckpt in ['2', '10001', '50001']:
    for objective_name, objective_config in [('dit', dit_config), ('vit', vit_config), ('gpt', gpt_config)]:
        # Note: Start with a resolution of [0.00001, 0.0001, 0.001, 0.01], then sweep within the bin.
        for lr in [0.00001, 0.00001 * 2.15, 0.00001 * 4.64, 0.0001, 0.0001 * 2.15, 0.0001 * 4.64, 0.001, 0.001 * 2.15, 0.001 * 4.64, 0.01]:
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label SPlusNoScaling --model.optimizer splus --model.do_scaling 0')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label Adam --model.optimizer adam')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label ScheduleFree --model.optimizer schedule_free')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label PSGD --model.optimizer psgd')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label Shampoo --model.optimizer shampoo --model.optimizer_freq 10')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label SOAP --model.optimizer soap')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label Sophia --model.optimizer sophia')

        # SPlus and Muon use a different LR scale, so we need higher LRs.
        for lr in [0.01, 0.01 * 2.15, 0.01 * 4.64, 0.1, 0.1 * 2.15, 0.1 * 4.64, 1.0, 1.0 * 2.15, 1.0 * 4.64, 10.0]:
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --label SPlus --model.lr {lr} --model.optimizer splus --model.do_scaling 1 --model.scaling_type avg {objective_config}')
            job_list.append(base + f' --load_dir gs://checkpoints/{objective_name}-step{ckpt} --wandb.name {objective_name.upper()}-C{ckpt} --model.lr {lr} {objective_config} --label Muon --model.optimizer muon')