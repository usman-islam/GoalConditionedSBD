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

#nvidia-debugdump -l
python3 -m rl.main --seed $seed --critic monolithic --env two-jaco-place-v0 --algo sac --gpu 0 --record 1 --num_record_samples 1 --prefix right-center-d --subdiv right_arm,right_hand,cube2-right_arm --env_args train_right-True/train_left-False --discriminator_loss_weight 10 --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-d.1/video --max_global_step 1000000
