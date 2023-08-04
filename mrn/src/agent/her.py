import copy
import numpy as np
import time
import torch

from mrn.src.model import *
from mrn.src.replay_buffer import ReplayBuffer
from mrn.src.utils import *
from mrn.src.sampler import Sampler
from mrn.src.agent.ddpg import DDPG


class HER(DDPG):
    """
    Hindsight Experience Replay agent
    """
    def __init__(self, args, env):
        super().__init__(args, env)
        self.sampler = Sampler(args, self.env.compute_reward_for_her)
        self.sample_func = self.sampler.sample_her_transitions
        self.buffer = ReplayBuffer(args, self.sample_func)
