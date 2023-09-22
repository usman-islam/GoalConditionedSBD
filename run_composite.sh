#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/%u/%j.out
#SBATCH --mem=50G

seed=12

#nvidia-debugdump -l
python3 -m rl.main --seed $seed --env two-jaco-pick-push-place-gc-v0 --record 0 --is_train 1 --gpu 0 --num_record_samples 1 --prefix ours --meta hard --subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm --subdiv_skills rl.two-jaco-push-v0.left-d.1/rl.two-jaco-pick-v0.right-d.1,rl.two-jaco-place-gc-v0.right-uniform-goals-d2.20 --primitive_subdiv left_arm,left_hand,cube1-left_arm/right_arm,right_hand,cube2-right_arm*right_arm,right_hand,cube2,goals-right_arm --max_meta_len 1 --max_global_step 2000000
