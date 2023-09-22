#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/%u/%j.out
#SBATCH --mem=50G
#SBATCH --exclude erc-hpc-comp037

seed=2

nvidia-debugdump -l
python -m rl.main --record 0 --is_train 1 --gpu 0 --env two-jaco-place-gc-v0 --prefix right-uniform-goals-d2 --seed $seed --subdiv right_arm,right_hand,cube2,goals-right_arm --env_args train_right-True/train_left-False/dest_center-False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-d.1/video --max_global_step 1200000 --wandb True --num_record_samples 1
