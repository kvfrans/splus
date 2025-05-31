from localutils.tpu_util import launch_tmux_jobs, launch_tmux_jobs_multi
import random

import os
os.system('sudo chmod -R 777 /nfs/jax-cache')

job_list = []
single_process = 'TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 '
debug_config = '--model.hidden_size 64 --model.depth 2 --model.num_heads 2 --model.mlp_ratio 1'
tiny_config = '--model.hidden_size 128 --model.depth 4 --model.num_heads 4 --model.mlp_ratio 4' 
xxsmall_config = '--model.hidden_size 72 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
xsmall_config = '--model.hidden_size 144 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
small_config = '--model.hidden_size 384 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
big_config = '--model.hidden_size 768 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
xlarge_config = '--model.hidden_size 1152 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4'
# large_config = '--model.hidden_size 1024 --model.depth 24 --model.num_heads 16 --model.mlp_ratio 4'
# xlarge_config = '--model.hidden_size 1152 --model.depth 28 --model.num_heads 16 --model.mlp_ratio 4'

dit_config = '--model.train_type dit --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1 --model.patch_size 2'
vit_config = '--model.train_type vit --dataset_name imagenet256-augment --model.patch_size 16 --model.use_stable_vae 0'
gpt_config = '--model.train_type gpt --dataset_name openwebtext --model.use_stable_vae 0'



base = f'python train.py --wandb.group Apr21-MainTable-10K --model.sharding fsdp --log_interval 10 --max_steps 10_000 --model.warmup 200 --model.weight_decay 0.1 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --model.sequence_length 256 --model.hidden_size 768 --batch_size 1024'
# job_list.append('ls')
# job_list.append('ls')
for ckpt in ['2']:
    # for lr in [0.1 * 2.15, 0.1 * 4.64, 1.0, 1.0 * 2.15]:
    # for lr in [0.001, 0.1]:
    # for lr in [0.001 * 2.15, 0.01 * 4.15, 0.1 * 2.15, 0.1 * 4.64]:
    # for lr in [0.05, 0.07, 0.1, 0.17]:
    for lr in [0.1, 0.2, 0.3, 0.5, 1.0]:
        job_list.append(base + f' --load_dir gs://rll-tpus-kvfrans/checkpoints/ranklearn/gpt-step{ckpt} --wandb.name GPT-C{ckpt}-{lr}-First {gpt_config} --label SPlus9 --model.lr {lr} --model.optimizer splus --model.do_scaling 1 --model.scaling_type first')
        job_list.append(base + f' --load_dir gs://rll-tpus-kvfrans/checkpoints/ranklearn/gpt-step{ckpt} --wandb.name GPT-C{ckpt}-{lr}-Avg {gpt_config} --label SPlus9 --model.lr {lr} --model.optimizer splus --model.do_scaling 1 --model.scaling_type avg')

    # for lr in [0.3, 0.5]:
        # job_list.append(base + f' --load_dir gs://rll-tpus-kvfrans/checkpoints/ranklearn/gpt-step{ckpt} --wandb.name GPT-C{ckpt}-{lr}-ExactHardcode {gpt_config} --label SPlus8 --model.lr {lr} --model.optimizer splus --model.do_scaling 1 --model.splus_constant 0.0005')
        # job_list.append(base + f' --load_dir gs://rll-tpus-kvfrans/checkpoints/ranklearn/gpt-step{ckpt} --wandb.name GPT-C{ckpt}-{lr}-SimpleConstantHardcode {gpt_config} --label SPlus8 --model.lr {lr} --model.optimizer splus --model.do_scaling 1 --model.splus_constant 0.001')

launch_tmux_jobs_multi(job_list, start_dir='/nfs/splus/')

# base = f'python train.py --wandb.group Apr21-MainTable-10K --model.sharding fsdp --log_interval 10 --max_steps 10_000 --model.warmup 200 --model.weight_decay 0.1 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --model.sequence_length 256 --model.hidden_size 768 --batch_size 1024'
# for ckpt in ['10001']:
#     for lr in [0.0001 * 2.15]:
#             job_list.append(base + f' --load_dir gs://rll-tpus-kvfrans/checkpoints/ranklearn/gpt-step{ckpt} --wandb.name GPT-C{ckpt}-{lr} --label AdamOirigin --model.lr {lr} --model.optimizer adam')
# launch_tmux_jobs(['a','b','c','d'], job_list, session_name="tpu3", start_dir='/nfs/ranklearn/')
