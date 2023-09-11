#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/%u/%j.out
#SBATCH --mem=50G

seed=1

nvidia-debugdump -l
python -m rl.main --record 1 --is_train 0 --env two-jaco-place-gc-v0 --prefix left-uniform-goals-d2 --seed $seed --subdiv left_arm,left_hand,cube2,goals-left_arm --env_args train_right-False/train_left-True/dest_center-False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.left-d.1/video --max_global_step 1000000 --wandb True --gpu 0 --num_record_samples 1
