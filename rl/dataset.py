from collections import defaultdict, OrderedDict
from time import time

import numpy as np
import torch


class ReplayBuffer:
    """ Replay Buffer. """

    def __init__(self, keys, buffer_size, sample_func):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    def store_episode(self, rollout):
        """ Stores the episode. """
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    def sample(self, batch_size):
        """ Samples the data from the replay buffer. """
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac'])


class RandomSampler:
    """ Samples a batch of transitions from replay buffer. """
    
    def __init__(self, ob_space, ac_space, reward_func, config):
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.reward_func = reward_func
        self.config = config
        self.relabel_rate = config.relabel_rate
    
    # MRN format to SBD format
    def _wrap_to_sbd_format(self, batch, wrapped_shape):
        dim_counter = 0
        wrapped_batch = OrderedDict()
        for item in wrapped_shape.items():
            sub_comp_len = item[1]
            wrapped_batch[item[0]] = batch[:, dim_counter : dim_counter + sub_comp_len]
            dim_counter += sub_comp_len
        return wrapped_batch
    
    def _unwrap_from_batch_format(self, batch, unwrapped_dim):
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

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch['ac'])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [np.random.randint(len(episode_batch['ac'][episode_idx])) for episode_idx in episode_idxs]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = \
                [episode_batch[key][episode_idx][t] for episode_idx, t in zip(episode_idxs, t_samples)]

        transitions['ob_next'] = [
            episode_batch['ob'][episode_idx][t + 1] for episode_idx, t in zip(episode_idxs, t_samples)]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions
    
    def sample_her_transitions(self, rollout, size):
        
        S = rollout['ob']
        A = rollout['ac']
        AG = rollout['achieved_goal']
        G = rollout['desired_goal']
        R = rollout['rew']
        info = rollout['info']
        done = rollout['done']
        
        # S = np.array(rollout['ob'], dtype=object)
        # A = np.array(rollout['ac'], dtype=object)
        # AG = np.array(rollout['achieved_goal'], dtype=np.ndarray)
        # G = np.array(rollout['desired_goal'], dtype=np.ndarray)
        # print('G:', G.shape)
        # raise Exception()
        # R = np.array(rollout['rew'], dtype=np.float32)
        # info = np.array(rollout['info'], dtype=object)
        
        # S = np.expand_dims(S, axis=0)
        # A = np.expand_dims(A, axis=0)
        # AG = np.expand_dims(AG, axis=0)
        # G = np.expand_dims(G, axis=0)
        # R = np.expand_dims(R, axis=0)
        # info = np.expand_dims(info, axis=0)
        
        # S: (batch, T+1)
        B = len(rollout['ac'])

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        # ep_lengths = np.zeros((size,), dtype=np.int32)
        # for i_ep in range(size):
        #     n_padding_zeros = 0
        #     for elem in np.flip(S[epi_idx[i_ep]]):
        #         if isinstance(elem, dict):
        #             break
        #         n_padding_zeros += 1
        #     ep_lengths[i_ep] = T - n_padding_zeros
        #     # ep_lengths[i_ep] = len(np.where(isinstance(S[epi_idx[i_ep]], dict))[0])
        ep_lengths = np.array([len(A[idx]) for idx in epi_idx], dtype=np.int32)
        t = np.random.randint(ep_lengths, size=size)
        # S_   =  S[epi_idx, t].copy() # (size, dim_state)
        # A_   =  A[epi_idx, t].copy()
        # AG_  = AG[epi_idx, t].copy()
        # G_   =  G[epi_idx, t].copy()
        # NS_  =  S[epi_idx, t+1].copy()
        # NAG_ = AG[epi_idx, t+1].copy()
        # info_ = info[epi_idx, t].copy()
        S_   =  np.array([ep[t[i]] for i, ep in enumerate(np.array(S)[epi_idx])], dtype=object)
        A_   =  np.array([ep[t[i]] for i, ep in enumerate(np.array(A)[epi_idx])], dtype=object)
        AG_  = np.array([ep[t[i]] for i, ep in enumerate(np.array(AG)[epi_idx])], dtype=np.ndarray)
        G_   =  np.array([ep[t[i]] for i, ep in enumerate(np.array(G)[epi_idx])], dtype=np.ndarray)
        NS_  =  np.array([ep[t[i]+1] for i, ep in enumerate(np.array(S)[epi_idx])], dtype=object)
        NAG_ = np.array([ep[t[i]+1] for i, ep in enumerate(np.array(AG)[epi_idx])], dtype=np.ndarray)
        info_ = np.array([ep[t[i]] for i, ep in enumerate(np.array(info)[epi_idx])], dtype=object)
        done_ = np.array([ep[t[i]] for i, ep in enumerate(np.array(done)[epi_idx])], dtype=bool)

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (ep_lengths - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        # her_AG = AG[epi_idx[her_idx], future_t]
        her_AG = np.array([ep[future_t[i]] for i, ep in enumerate(np.array(AG)[epi_idx[her_idx]])], dtype=np.ndarray)

        #tt = self.get_closest_goal_state(AG[epi_idx[her_idx]], t[her_idx], future_t)
        #GS = S_.copy()
        #GS[her_idx] = S[epi_idx[her_idx], tt].copy()
        mask = np.zeros((size,))
        mask[her_idx] = 1.0

        G_[her_idx] = her_AG
        # prev_S = S[epi_idx, np.maximum(t-1, 0)].copy()
        prev_S = np.array([ep[np.maximum(t[i]-1, 0)] for i, ep in enumerate(np.array(S)[epi_idx])], dtype=object)
        rew, _, _ = self.reward_func(NAG_, G_, None, prev_S, S_, info_)
        R_ = np.expand_dims(rew, 1) # (size, 1)
        
        transition = {
            'ob' : self._wrap_to_sbd_format(self._unwrap_from_batch_format(S_, self.config.dim_state), self.ob_space),
            'ob_next' : self._wrap_to_sbd_format(self._unwrap_from_batch_format(NS_, self.config.dim_state), self.ob_space),
            'ac' : self._wrap_to_sbd_format(self._unwrap_from_batch_format(A_, self.config.dim_action), self.ac_space.shape),
            'desired_goal' : G_,
            'rew' : R_,
            'done': done_
        }
        return transition

