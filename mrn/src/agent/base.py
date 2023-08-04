import copy
import numpy as np
import time
import torch

from mrn.src.model import *
from mrn.src.replay_buffer import ReplayBuffer
from mrn.src.utils import *
from mrn.src.sampler import Sampler

from rl.policies.mlp_actor_critic import MlpActor

import copy
from collections import OrderedDict

################################################################################
#
# Agent class for sparse reward RL
#
################################################################################


class Agent(object):
    """
    The sparse reward RL agent superclass
    """
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.best_success_rate = 0.0

        # self.actor = Actor(args)
        self.actor = MlpActor(args, self.env.ob_space, self.env.ac_space, args.tanh_policy)
        
        # Hack to comply with SBD rollouts.py
        self._actors = [[self.actor]]

        sync_networks(self.actor)

        if self.args.cuda:
            self.actor.cuda()

        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),
                                             lr=self.args.lr_actor)

        # # observation/goal normalizer
        # self.s_norm = Normalizer(size=args.dim_state,
        #                          default_clip_range=self.args.clip_range)
        # self.g_norm = Normalizer(size=args.dim_goal,
        #                          default_clip_range=self.args.clip_range)
        self.sampler = Sampler(args, self.env.compute_reward)
        self.sample_func = self.sampler.sample_ddpg_transitions
        
        self.s_norm = Normalizer(size=args.dim_state,
                                 default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(size=args.dim_goal,
                                 default_clip_range=self.args.clip_range)

    def _preproc_inputs(self, s=None, g=None, unsqueeze=False):
        s_tensor = None
        if s is not None:
            s_ = self.s_norm.normalize(
                    np.clip(s, -self.args.clip_obs, self.args.clip_obs))
            s_tensor = numpy2torch(s_, unsqueeze=unsqueeze, cuda=self.args.cuda)

        g_tensor = None
        if g is not None:
            g_ = self.g_norm.normalize(
                    np.clip(g, -self.args.clip_obs, self.args.clip_obs))
            g_tensor = numpy2torch(g_, unsqueeze=unsqueeze, cuda=self.args.cuda)
        return s_tensor, g_tensor

    def _select_actions(self, s_tensor, g_tensor, stochastic):
        with torch.no_grad():
            a = self.actor(s_tensor, g_tensor).detach().cpu().squeeze().numpy()

        if not stochastic:
            return a

        # gaussian noise
        max_action = self.args.max_action
        a += self.args.noise_eps * max_action * np.random.randn(*a.shape)
        a = np.clip(a, -max_action, max_action)

        # eps-greedy
        rand_a = np.random.uniform(low=-max_action, high=max_action,
                                   size=self.args.dim_action)

        # choose if use the random actions
        a += np.random.binomial(1, self.args.random_eps, 1)[0] * (rand_a - a)
        return a

    def _update_normalizer(self, S, A, AG, G, R, info):
        transition = self.sample_func(
                S, A, AG, G, R, A.shape[1], info)

        S_ = self._unwrap_from_buffer_format(transition['S'], self.args.dim_state)
        G_ = transition['G']

        S_ = np.clip(S_, -self.args.clip_obs, self.args.clip_obs)
        G_ = np.clip(G_, -self.args.clip_obs, self.args.clip_obs)
        
        self.s_norm.update(S_)
        self.g_norm.update(G_)

        self.s_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            d = self.args.polyak
            tp.data.copy_( (1-d) * sp.data + d * tp.data )

    def _update(self):
        pass
    
    # Replay buffer batch format to MRN format
    def _unwrap_from_buffer_format(self, batch, unwrapped_dim):
        if isinstance(list(batch[0].items())[0][1], torch.Tensor):
            unwrapped_batch = torch.zeros((len(batch), unwrapped_dim), dtype=torch.float32, device=self.args.device)
        else:
            unwrapped_batch = np.zeros((len(batch), unwrapped_dim), dtype=np.float32)
        for i, ob in enumerate(batch):
            dim_counter = 0
            for item in ob.items():
                sub_comp = item[1]
                unwrapped_batch[i, dim_counter : dim_counter + len(sub_comp)] = sub_comp
                dim_counter += len(sub_comp)
        return unwrapped_batch
    
    # SBD format to MRN format
    def _unwrap_from_sbd_format(self, batch, unwrapped_dim):
        if isinstance(list(batch.items())[0][1], torch.Tensor):
            unwrapped_batch = torch.zeros((len(list(batch.items())[0][1]), unwrapped_dim), dtype=torch.float32, device=self.args.device)
        else:
            unwrapped_batch = np.zeros((len(list(batch.items())[0][1]), unwrapped_dim), dtype=np.float32)
        dim_counter = 0
        for item in batch.items():
            sub_comp = item[1]
            unwrapped_batch[:, dim_counter : dim_counter + len(sub_comp)] = sub_comp
            dim_counter += len(sub_comp)
        return unwrapped_batch
    
    # MRN format to SBD format
    def _wrap_to_sbd_format(self, batch, wrapped_shape):
        dim_counter = 0
        wrapped_batch = OrderedDict()
        for item in wrapped_shape.items():
            sub_comp_len = item[1]
            wrapped_batch[item[0]] = batch[:, dim_counter : dim_counter + sub_comp_len]
            dim_counter += sub_comp_len
        return wrapped_batch
    
    # Added for SBD. Note, this assumes wrapped observation input
    def act(self, ob, is_train=True):
        """ Returns action and the actor's activation given an observation @ob. """
        ob_, _ = self._preproc_inputs(
            s=self._unwrap_from_buffer_format(
                np.array([ob], dtype=object),
                self.args.dim_state
            )
        )
        ob = self._wrap_to_sbd_format(
            ob_,
            self.env.ob_space
        )
        if hasattr(self, '_actor'):
            ac, activation = self._actor.act(ob, is_train=is_train)
        else:
            ac, activation = self._actors[0][0].act(ob, is_train=is_train)

        return ac, activation

    def select_action(self, o, stochastic=True):
        s  = o['observation'].astype(np.float32)
        g  = o['desired_goal'].astype(np.float32)
        s_tensor, g_tensor = self._preproc_inputs(s, g, unsqueeze=True)
        return self._select_actions(s_tensor, g_tensor, stochastic)

    def prefill_buffer(self):
        if self.args.n_init_episodes == 0:
            return
        n_threads = MPI.COMM_WORLD.Get_size()
        n_collects = (self.args.n_init_episodes // self.args.rollout_n_episodes // n_threads+1)
        for _ in range(n_collects):
            (S, A, AG, G, info), _ = self.collect_rollout(uniform_random_action=True)
            self.buffer.store_episode(S, A, AG, G, info)

    def collect_rollout(self, uniform_random_action=False, stochastic=True):
        n_episodes = self.args.rollout_n_episodes
        dim_state  = self.args.dim_state
        dim_action = self.args.dim_action
        dim_goal   = self.args.dim_goal
        T = self.args.max_episode_steps
        max_action = self.args.max_action

        S  = np.empty([n_episodes, T+1]).astype(object)
        A  = np.empty([n_episodes, T]).astype(object)
        AG = np.empty([n_episodes, T+1, dim_goal]).astype(np.float32)
        G  = np.empty([n_episodes, T, dim_goal]).astype(np.float32)
        info_mat  = np.empty([n_episodes, T]).astype(object)
        success = np.zeros((n_episodes), np.float32)

        for i in range(n_episodes):
            o = self.env.reset()
            for t in range(T):
                if uniform_random_action:
                    a = np.random.uniform(low=-max_action,
                                          high=max_action,
                                          size=(dim_action,))
                else:
                    a = self.select_action(o, stochastic=stochastic)
                new_o, r, d, info = self.env.step(a)
                S[i][t]  = copy.deepcopy(o['observation'])
                AG[i][t] = o['achieved_goal'].copy()
                G[i][t]  = o['desired_goal'].copy()
                A[i][t]  = copy.deepcopy(a)
                info_mat[i][t] = copy.deepcopy(info)
                o = new_o

            success[i] = info['success']
            S[i][t+1] = o['observation'].copy()
            AG[i][t+1] = o['achieved_goal'].copy()
        return (S, A, AG, G, info_mat), success

    def eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.eval_rollout_n_episodes):
            per_success_rate = []
            o = self.env.reset()
            for _ in range(self.args.max_episode_steps):
                with torch.no_grad():
                    a = self.select_action(o, stochastic=False)
                o, _, _, info = self.env.step(a)
                per_success_rate.append(info['success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        local_hitting_time = first_nonzero(total_success_rate, 1, self.args.max_episode_steps+1).mean()
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        global_hitting_time = MPI.COMM_WORLD.allreduce(local_hitting_time, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), global_hitting_time / MPI.COMM_WORLD.Get_size()

    def save_model(self, stats):
        sd = {
            "actor": self.actor.state_dict(),
            "args": self.args,
            "stats": stats,
            "s_mean": self.s_norm.mean, 
            "s_std": self.s_norm.std,
            "g_mean": self.g_norm.mean,
            "g_std": self.g_norm.std,
        }
        if hasattr(self, "critic"):
            sd["critic"] = self.critic.state_dict()

        torch.save(sd,
            os.path.join(self.args.save_dir, f"{self.args.experiment_name}.pt"))
        if stats['successes'][-1] > self.best_success_rate:
            self.best_success_rate = stats['successes'][-1]
            torch.save(sd,
                os.path.join(self.args.save_dir, f"{self.args.experiment_name}_best.pt"))

    def learn(self):
        pass
