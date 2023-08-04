""" Launch RL training and evaluation. """

import sys
import signal
import os
import json

import numpy as np
import torch
from six.moves import shlex_quote
from mpi4py import MPI

from rl.config import argparser
from rl.trainer import Trainer
from util.logger import logger

from datetime import datetime


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def run(config):
    """
    Runs Trainer.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    config.rank = rank
    config.is_chef = rank == 0
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size()
    
    config.goal_dim = 3

    if config.is_chef:
        logger.warn('Run a base worker.')
        make_log_files(config)
    else:
        logger.warn('Run worker %d and disable logger.', config.rank)
        import logging
        logger.setLevel(logging.CRITICAL)

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.virtual_display is not None:
        os.environ["DISPLAY"] = config.virtual_display

    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
        config.cuda = True
    else:
        config.device = torch.device("cpu")

    # build a trainer
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        trainer.evaluate()
        logger.info("Finish evaluating")


def make_log_files(config):
    """
    Sets up log directories and saves git diff and command line.
    """
    log_dir_name = 'rl.{}.{}.{}'.format(config.env, config.prefix, config.seed)
    config.run_name = '{}.{}'.format(log_dir_name, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    # config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    config.log_dir = os.path.join(config.log_root_dir, log_dir_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    if config.is_train:
        config.record_dir = os.path.join(config.log_dir, 'video')
        config.results_progress_dir = os.path.join(config.log_dir, 'progress', f'seed_{config.seed}')
    else:
        config.record_dir = os.path.join(config.log_dir, 'eval_video')
        config.results_progress_dir = os.path.join(config.log_dir, 'eval_progress', f'seed_{config.seed}')
    config.plot_dir = os.path.join(config.results_progress_dir, 'plots')
    logger.info('Create video directory: %s', config.record_dir)
    os.makedirs(config.record_dir, exist_ok=True)
    os.makedirs(config.results_progress_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)

    if config.subdiv_skill_dir is None:
        config.subdiv_skill_dir = config.log_root_dir

    if config.is_train:
        # log git diff
        cmds = [
            "echo `git rev-parse HEAD` >> {}/git.txt".format(config.log_dir),
            "git diff >> {}/git.txt".format(config.log_dir),
            "echo 'python -m rl.main {}' >> {}/cmd.sh".format(
                ' '.join([shlex_quote(arg) for arg in sys.argv[1:]]),
                config.log_dir),
        ]
        os.system("\n".join(cmds))

        # log config
        param_path = os.path.join(config.log_dir, 'params.json')
        logger.info('Store parameters in %s', param_path)
        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    args, unparsed = argparser()
    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        run(args)

