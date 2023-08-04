# Goal-conditioned SBD

This is a work-in-progress adjustment to the Skill Behaviour Diversification (SBD) code found here: https://github.com/clvrai/coordination. The idea is to use a goal-conditioned RL (GCRL) method to train both the primitive skill policies and the meta-policy so that the goals in the tested environments can vary from episode to episode. For example, in the PICK-PUSH-PLACE environment, the goal is the coordinate where the arms need to both push the tray and place the cube. In the SBD code, this is a hardcoded coordinate but we wish to vary this during training. We use Metric Residual Networks (MRN) as the GCRL method and its code can be found here: https://github.com/Cranial-XIX/metric-residual-network.

Before running the code, follow the installation instructions on https://github.com/clvrai/coordination and update lines 113 and 115 in rl/trainer.py with your own wandb information. Also, in order to quickly sync the folder to a remote server (to run on a cluster), you can use the script rs.sh, which uses rsync. Just install rsync and update the script with your own host and directory location on the host.

So far MRN is being implemented and tested on the PLACE primitive, which requires the PICK primitive to have already run successfully. Run PICK by executing
```
./run_pick_right-d.sh
```
I recommend doing this on a cluster as it can take a few hours. If your machine does not have a cuda-enabled GPU, remove the --gpu 0 flag. The scripts are meant to be run by SLURM sbatch and they contain appropriate run commands taken from the SBD README.

To train on the PLACE primitive once PICK has completed, run

```
./run_place_right-center-d.sh
```

This will use SAC. If you want to use MRN instead, change the algo parameter in run_place_right-center-d.sh to mrn. To view results, go into log/rl.two-jaco-place-v0.right-center-d.1/wandb and run
```
wandb sync
```
then have a look at your wandb project. To view the result videos, go to log/rl.two-jaco-place-v0.right-center-d.1/video/. Note, the 1 refers to the seed, which can be changed in the run_place_right-center-d.sh script.
