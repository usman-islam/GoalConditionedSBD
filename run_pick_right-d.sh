#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/%u/%j.out
#SBATCH --mem=50G


#nvidia-debugdump -l
python3 -m rl.main --seed 1 --env two-jaco-pick-v0 --record 1 --gpu 0 --num_record_samples 1 --prefix right-d --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --discriminator_loss_weight 10 --save_qpos True --max_global_step 300000
