python3 -m rl.main --env two-jaco-place-gc-v0 --prefix right-side-two-goals-d1 --seed 40 --subdiv right_arm,right_hand,cube2,goals-right_arm --env_args train_right-True/train_left-False/dest_center-False --save_qpos True --init_qpos_dir log/rl.two-jaco-pick-v0.right-d.1/video --max_global_step 800000 --gpu 0 --wandb True