from collections import OrderedDict

import numpy as np
import gym

import pettingzoo_1_12_0.pettingzoo.mpe.simple_v2 as simple_v2

class MPESimpleEnv(gym.Env):
    def __init__(self, **kwargs):
        self.name = 'mpe-simple'
        self.x_low = -1.
        self.y_low = -1.
        self.x_high = 1.
        self.y_high = 1.
        
        self._env_config = {}

        # config
        self._env_config.update({
            'max_cycles': 25,
            'continuous_actions': True
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })
        
        self.env = simple_v2.env(max_cycles=self._env_config['max_cycles'], continuous_actions=self._env_config['continuous_actions'])

    def compute_reward(self, new_achieved_goals, goals, c, prev_obs, new_obs, transition_infos, prev_transition_infos):
        
        batch_size = goals.shape[0]
        rewards = []
        
        for transition in range(batch_size):
            info = transition_infos[transition]
            agent_p_pos = info['agent_0']['agent_p_pos']
            goal = goals[transition]
            
            rewards.append(-np.sum(np.square(goal - agent_p_pos)))
    
    def subdiv_ob(self, ob):
        return OrderedDict([('agent_0', ob)])
    
    def add_goals(self, ob):
        obs_and_goals = {'observation': ob}
        obs_and_goals['achieved_goal'] = np.array(self.current_position, dtype=np.float32)
        obs_and_goals['desired_goal'] = np.array(self.goal, dtype=np.float32)
        return obs_and_goals

    def step(self, a):
        new_ob, reward, done, info = self.env.step(a['agent_0'])
        self.current_position = self.env.world.agents[0].state.p_pos
        return self.add_goals(self.subdiv_ob(new_ob)), reward, done, info

    def reset(self):
        self.env.reset()
        new_goal = [np.random.uniform(self.x_low, self.x_high), np.random.uniform(self.y_low, self.y_high)]
        ob = [np.random.uniform(self.x_low, self.x_high), np.random.uniform(self.y_low, self.y_high)]
        
        # new_goal1 = [0., 0.1]
        # new_goal2 = [0.3, -0.02]
        # new_goal = new_goal1 if np.random.rand() > 0.5 else new_goal2
        
        self.current_position = ob
        self.goal = new_goal
        self.env.world.landmarks[0].state.p_pos = new_goal
        self.env.world.agents[0].state.p_pos = ob
