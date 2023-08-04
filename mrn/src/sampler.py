import numpy as np


class Sampler(object):
    """
    Helper class to sample transitions for learning.
    Methods like sample_her_transitions will relabel part of trajectories.
    """
    def __init__(self, args, env_reward_func):
        self.relabel_rate = args.relabel_rate
        # plus = 1.0 if not args.negative_reward else 0.0
        # make reward to {-1, 0} instead of {0, 1} if negative reward
        # self.reward_func = lambda ag, g, c: env_reward_func(ag, g, c) + plus
        self.reward_func = env_reward_func

        if args.negative_reward:
            self.achieved_func = lambda ag, g: env_reward_func(ag, g, None) + 1.0
        else:
            self.achieved_func = lambda ag, g: env_reward_func(ag, g, None)

        self.global_threshold = 80
        
        self.counter = 0

    def get_closest_goal_state(self, G, t1, t2):
        # S: (batch, T+1, dim_state)
        # G: (batch, T+1, dim_goal)
        # t1: (batch,) the starting state timestep
        # t2: (batch,) the goal state timestep
        # return: max_{t1 <= t <= t2} phi(S_t) = G_t2
        ts = []
        for i in range(G.shape[0]):
            t = t2[i]
            while self.achieved_func(G[i, t], G[i, t2[i]]) > 0.5 and t > t1[i]:
                t -= 1
            ts.append(t+1)
        return np.array(ts).astype(np.int32)

    def sample_ddpg_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
        }
        return transition
    
    def sample_ddpg_transitions_rew(self, S, A, AG, G, R, size, info, done):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        epi_idx = np.arange(B - size, B).tolist()
        # if self.counter == 1:
        #     print('1')
        #     epi_idx = np.arange(1,2).tolist()
        # else:
        #     print('0')
        #     epi_idx = np.arange(0,1).tolist()
        # self.counter += 1
        
        ep_lengths = np.zeros((size,), dtype=np.int32)
        for i_ep in range(size):
            n_padding_zeros = 0
            for elem in np.flip(S[epi_idx[i_ep]]):
                if isinstance(elem, dict):
                    break
                n_padding_zeros += 1
            ep_lengths[i_ep] = T - n_padding_zeros
            
        t = np.random.randint(np.ones_like(ep_lengths), ep_lengths, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        r_   =  R[epi_idx, t].copy()
        info_ = info[epi_idx, t].copy()
        done_ = done[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()
        
        # prev_S = S[epi_idx, np.maximum(t-1, 0)].copy()
        # prev_info = info[epi_idx, np.maximum(t-1, 0)].copy()
        # rew, _, _ = self.reward_func(NAG_, G_, None, prev_S, S_, info_, prev_info)
        # R_ = np.expand_dims(rew, 1) # (size, 1)
        
        prev_r = R[epi_idx, np.maximum(t-1, 0)].copy()
        prev_S = S[epi_idx, np.maximum(t-1, 0)].copy()
        prev_prev_S = S[epi_idx, np.maximum(t-2, 0)].copy()
        prev_NAG = AG[epi_idx, np.maximum(t, 0)].copy()
        prev_G = G[epi_idx, np.maximum(t-1, 0)].copy()
        prev_info = info[epi_idx, np.maximum(t-1, 0)].copy()
        prev_prev_info = info[epi_idx, np.maximum(t-2, 0)].copy()
        rew = self.reward_func(NAG_, G_, None, prev_S, S_, info_, prev_info)
        prev_rew = self.reward_func(prev_NAG, prev_G, None, prev_prev_S, prev_S, prev_info, prev_prev_info)
        reward_ctrl = np.array([elem['reward_ctrl'] for elem in info_], dtype=np.float)
        new_rew = rew - prev_rew + reward_ctrl
        
        new_rew = rew
        
        R_ = np.expand_dims(new_rew, 1) # (size, 1)
        prev_R = np.expand_dims(prev_rew, 1) # (size, 1)
        # transitions = self._buffer.sample(256)
        # print('store ep sample R:', transitions['R'])
        rdiff = r_ - R_
        prev_rdiff = prev_r - prev_R
        
        rdiff = r_ - R_
        # print('r_:', r_)
        # print('R_:', R_)
        # print('r - R:', rdiff)
        # raise Exception()

        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            # 'R' : r_,
            'done': done_,
            'NG': NAG_,
        }
        return transition

    def sample_her_transitions(self, S, A, AG, G, R, size, info, done):
        # S: (batch, T+1)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        ep_lengths = np.zeros((size,), dtype=np.int32)
        for i_ep in range(size):
            n_padding_zeros = 0
            for elem in np.flip(S[epi_idx[i_ep]]):
                if isinstance(elem, dict):
                    break
                n_padding_zeros += 1
            ep_lengths[i_ep] = T - n_padding_zeros
            # ep_lengths[i_ep] = len(np.where(isinstance(S[epi_idx[i_ep]], dict))[0])

        t = np.random.randint(ep_lengths, size=size)
        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()
        info_ = info[epi_idx, t].copy()
        done_ = done[epi_idx, t].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        #tt = self.get_closest_goal_state(AG[epi_idx[her_idx]], t[her_idx], future_t)
        #GS = S_.copy()
        #GS[her_idx] = S[epi_idx[her_idx], tt].copy()
        mask = np.zeros((size,))
        mask[her_idx] = 1.0

        G_[her_idx] = her_AG
        prev_S = S[epi_idx, np.maximum(t-1, 0)].copy()
        rew = self.reward_func(NAG_, G_, None, prev_S, S_, info_)
        R_ = np.expand_dims(rew, 1) # (size, 1)
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'NG': NAG_,
            'done': done_,
            #'GS': GS,
            'mask': mask
        }
        return transition

    def sample_mher_transitions(self, S, A, AG, G, size, get_imaginary_rollout, goal_idx):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        S_list = get_imaginary_rollout(S_, G_) # (size, n_steps, dim_state)

        idx = np.where(np.random.uniform(size=size) < 1.0)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[idx]
        future_AG = AG[epi_idx[idx], future_t]

        # imaginary relabel
        relabel_idx = (np.random.uniform(size=size) < 0.8)
        step_idx = np.random.randint(S_list.shape[1], size=size)
        last_state = S_list[np.arange(size), step_idx] # (size, dim_state)
        imaginary_goal = last_state[..., goal_idx.numpy()][relabel_idx]
        G_[relabel_idx] = imaginary_goal

        her_idx = (np.random.uniform(size=size) < 0.4)
        G_[relabel_idx & her_idx] = future_AG[relabel_idx & her_idx]
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        transition = {
            'S'  : S_,
            'NS' : NS_,
            'A'  : A_,
            'G'  : G_,
            'R'  : R_,
        }
        return transition

    def sample_gcsl_transitions(self, S, A, AG, G, size):
        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        G_[her_idx] = her_AG
        transition = {
            'S' : S_,
            'A' : A_,
            'G' : G_,
        }
        return transition

    def sample_wgcsl_transitions(self, S, A, AG, G, size,
                                 args, q_func, r_func, advque):

        # S: (batch, T+1, dim_state)
        B, T = A.shape[:2]

        # sample size episodes from batch
        epi_idx = np.random.randint(0, B, size)
        t = np.random.randint(T, size=size)

        S_   =  S[epi_idx, t].copy() # (size, dim_state)
        A_   =  A[epi_idx, t].copy()
        AG_  = AG[epi_idx, t].copy()
        G_   =  G[epi_idx, t].copy()
        NS_  =  S[epi_idx, t+1].copy()
        NAG_ = AG[epi_idx, t+1].copy()

        # determine which time step to sample
        her_idx = np.where(np.random.uniform(size=size) < self.relabel_rate)
        future_offset = (np.random.uniform(size=size) * (T - t)).astype(int)
        future_t = (t + 1 + future_offset)[her_idx]
        her_AG = AG[epi_idx[her_idx], future_t]

        G_[her_idx] = her_AG
        R_ = np.expand_dims(self.reward_func(NAG_, G_, None), 1) # (size, 1)

        W_ = pow(args.gamma, future_offset).reshape(-1, 1)
        adv = args.gamma * q_func(NS_, G_) - \
                q_func(S_, G_) + R_
        advque.update(adv)
        self.global_threshold = min(self.global_threshold + \
                args.wgcsl_baw_delta, args.wgcsl_baw_max)
        threshold = advque.get(self.global_threshold)

        W_ *= np.clip(np.exp(adv), 0, args.wgcsl_adv_clip)

        positive = adv.copy()
        positive[adv >= threshold] = 1
        positive[adv < threshold] = 0.05
        W_ *= positive
        transition = {
            'S' : S_,
            'NS': NS_,
            'A' : A_,
            'G' : G_,
            'R' : R_,
            'W' : W_,
        }
        return transition
