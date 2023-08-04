import os
import time
import h5py
from collections import OrderedDict

import numpy as np

from env.jaco.two_jaco import TwoJacoEnv
from env.transform_utils import quat_dist

import mujoco_py
import copy


class TwoJacoPlaceGCEnv(TwoJacoEnv):
    def __init__(self, **kwargs):
        self.name = 'two-jaco-place-gc'
        super().__init__('two_jaco_pick.xml', **kwargs)

        # config
        self._env_config.update({
            "train_left": True,
            "train_right": True,
            "success_reward": 500,
            "target_xy_reward": 500,
            "target_z_reward": 500,
            "move_finish_reward": 50,
            "grasp_reward": 100,
            "inair_reward": 0,
            "init_randomness": 0.005,
            "dest_pos": [0.3, -0.02, 0.86],
            # "dest_center": True, # set destination to center
            "ctrl_reward": 1e-4,
            "max_episode_steps": 100,
            "init_qpos_dir": None
        })
        self._env_config.update({ k:v for k,v in kwargs.items() if k in self._env_config })

        # # target position
        # if not self._env_config['dest_center']:
        #     if self._env_config['train_left'] and not self._env_config['train_right']:
        #         self._env_config['dest_pos'] = [0.05, 0.2, 0.86] # 0.15
        #     elif not self._env_config['train_left'] and self._env_config['train_right']:
        #         self._env_config['dest_pos'] = [0.05, -0.2, 0.86] # 0.15

        # state
        self._hold_duration = [0, 0]
        self._box_z = [0, 0]
        self._min_height = self._env_config['dest_pos'][-1] + 0.08

        self._get_reference()

    def _get_reference(self):
        self.cube_body_id = [self.sim.model.body_name2id("cube{}".format(i)) for i in [1, 2]]
        self.cube_geom_id = [self.sim.model.geom_name2id("cube{}".format(i)) for i in [1, 2]]
        
# gripper centers

    def compute_reward(self):
        ''' Environment reward consists of place reward, xy-position reward
            Place reward = -|current z position - expected z position|
            Xy-position reward = -||current xy position - init xy position||^2
        '''
        done = False
        off_table = [False, False]
        info = {}

        # compute gripper centers
        cube_z = [self._get_pos("cube1")[-1], self._get_pos("cube2")[-1]]
        gripper_centers = self._get_gripper_centers()
        

        # get init box pos
        if self._t == 0:
            self._init_box_pos = [self._get_pos('cube{}'.format(i + 1)) for i in range(2)]

        # object placing reward before placing is finished
        in_hand = [True, True]
        target_xy_rewards = [0, 0]
        target_z_rewards = [0, 0]
        target_dist_xy = [0, 0]
        target_dist_z = [0, 0]
        grasp_count = [0, 0]
        grasp_rewards = [0, 0]
        inair_rewards = [1, 1]
        move_finish_rewards = [0, 0]
        for i in range(2):
            dist_cube_hand = np.linalg.norm(self._get_pos('cube{}'.format(i + 1)) - gripper_centers[i])
            in_hand[i] = dist_cube_hand < 0.08
            off_table[i] = cube_z[i] < 0.8

            # place reward
            cube_pos = self._get_pos('cube{}'.format(i + 1))
            target_dist_xy[i] = float(np.linalg.norm(cube_pos[:2] - self._env_config['dest_pos'][:2]))
            target_dist_z[i] = float(np.abs(cube_pos[-1] - self._env_config['dest_pos'][-1]))
            if self._stage[i] == 'move':
                height_decrease = self._init_box_pos[i][-1] - cube_pos[-1]
                target_z_rewards[i] = -self._env_config["target_z_reward"] * max(0, height_decrease)
                if target_dist_xy[i] < 0.06 and cube_pos[-1] > self._min_height:
                    move_finish_rewards[i] += self._env_config['move_finish_reward']
                    self._stage[i] = 'place'
            if self._stage[i] == 'place':
                target_z_rewards[i] = -self._env_config["target_z_reward"] * target_dist_z[i]
            target_xy_rewards[i] -= self._env_config["target_xy_reward"] * target_dist_xy[i]

            # grasp reward
            contact_flag = [False] * 3
            geom_name_prefix = 'jaco_{}_link_finger'.format('l' if i == 0 else 'r')
            for j in range(self.sim.data.ncon):
                c = self.sim.data.contact[j]
                for k in range(3):
                    geom_name = '{}_{}'.format(geom_name_prefix, k + 1)
                    geom_id = self.sim.model.geom_name2id(geom_name)
                    if c.geom1 == geom_id and c.geom2 == self.cube_geom_id[i]:
                        contact_flag[k] = True
                    if c.geom2 == geom_id and c.geom1 == self.cube_geom_id[i]:
                        contact_flag[k] = True
            grasp_count[i] = np.array(contact_flag).astype(int).sum()
            if self._t == 0:
                self._init_grasp_count[i] = grasp_count[i]
            grasp_rewards[i] = -self._env_config["grasp_reward"] * max(0, self._init_grasp_count[i] - grasp_count[i])

            # in air reward
            table_geom_id = self.sim.model.geom_name2id("table_collision")
            for j in range(self.sim.data.ncon):
                c = self.sim.data.contact[j]
                if c.geom1 == table_geom_id and c.geom2 == self.cube_geom_id[i]:
                    inair_rewards[i] = 0
                if c.geom2 == table_geom_id and c.geom1 == self.cube_geom_id[i]:
                    inair_rewards[i] = 0
            if target_dist_xy[i] < 0.06:
                inair_rewards[i] = 1
            inair_rewards[i] *= self._env_config["inair_reward"]

        # success criteria
        self._success = True
        if self._env_config['train_left']:
            self._success &= in_hand[0] and target_dist_xy[0] < 0.04 and target_dist_z[0] < 0.03 and self._stage[0] == 'place'
        if self._env_config['train_right']:
            self._success &= in_hand[1] and target_dist_xy[1] < 0.04 and target_dist_z[1] < 0.03 and self._stage[1] == 'place'

        # success reward
        success_reward = 0
        if self._success:
            print('All places success!')
            success_reward = self._env_config["success_reward"]

        done = self._success
        if self._env_config['train_left']:
            done |= (not in_hand[0])
        if self._env_config['train_right']:
            done |= (not in_hand[1])

        reward = success_reward 
        in_hand_rewards = [0, 0]
        if self._env_config["train_left"]:
            done |= off_table[0]
            reward += target_xy_rewards[0] + target_z_rewards[0] + grasp_rewards[0] + inair_rewards[0] + move_finish_rewards[0]
        if self._env_config["train_right"]:
            done |= off_table[1]
            reward += target_xy_rewards[1] + target_z_rewards[1] + grasp_rewards[1] + inair_rewards[1] + move_finish_rewards[1]
            
        # if self._env_config["train_left"]:
        #     done |= off_table[0]
        #     # reward += target_xy_rewards[0] + target_z_rewards[0] + grasp_rewards[0] + inair_rewards[0] + move_finish_rewards[0]
        #     # reward = target_xy_rewards[0]
        #     # reward = target_z_rewards[0]
        #     # reward = grasp_rewards[0]
        #     # reward = inair_rewards[0]
        #     reward = move_finish_rewards[0]
        # if self._env_config["train_right"]:
        #     done |= off_table[1]
        #     # reward += target_xy_rewards[1] + target_z_rewards[1] + grasp_rewards[1] + inair_rewards[1] + move_finish_rewards[1]
        #     # reward = target_xy_rewards[1]
        #     # reward = target_z_rewards[1]
        #     # reward = grasp_rewards[1]
        #     # reward = inair_rewards[1]
        #     reward = move_finish_rewards[1]
            
        # reward = self._env_config['dest_pos'][0]
        
        # print('gc target_xy_rewards:', target_xy_rewards)
        # print('gc target_z_rewards:', target_z_rewards)
        # print('gc grasp_rewards:', grasp_rewards)
        # print('gc inair_rewards:', inair_rewards)
        # print('gc move_finish_rewards:', move_finish_rewards)
        # print('gc reward:', reward)

        info = {"reward_xy_1": target_xy_rewards[0],
                "reward_xy_2": target_xy_rewards[1],
                "reward_z_1": target_z_rewards[0],
                "reward_z_2": target_z_rewards[1],
                "reward_grasp_1": grasp_rewards[0],
                "reward_grasp_2": grasp_rewards[1],
                "reward_inair_1": inair_rewards[0],
                "reward_inair_2": inair_rewards[1],
                "reward_move_finish_1": move_finish_rewards[0],
                "reward_move_finish_2": move_finish_rewards[1],
                "grasp_count_1": grasp_count[0],
                "grasp_count_2": grasp_count[1],
                "in_hand": in_hand,
                "target_pos": self._env_config['dest_pos'],
                "cube1_pos": self._get_pos("cube1"),
                "cube2_pos": self._get_pos("cube2"),
                "gripper1_pos": gripper_centers[0],
                "gripper2_pos": gripper_centers[1],
                "target_dist_xy": np.round(target_dist_xy, 3),
                "target_dist_z": np.round(target_dist_z, 3),
                "curr_qpos_l": np.round(self.data.qpos[1:10], 1).tolist(),
                "curr_qpos_r": np.round(self.data.qpos[10:19], 1).tolist(),
                "stage": self._stage,
                "success": self._success,
                
                "init_box_pos": self._init_box_pos,
                "env_config": self._env_config,
                "min_height": self._min_height,
                "sim_data_qpos": copy.deepcopy(self.sim.data.qpos),
                "cube_geom_id": self.cube_geom_id,
                "init_grasp_count": self._init_grasp_count,
                "gripper_centers": gripper_centers,
                "grasp_rewards": grasp_rewards,
                "inair_rewards": inair_rewards
                
                
        }

        return reward, done, info
    
    def compute_reward_for_her(self, new_achieved_goals, goals, c, prev_obs, new_obs, transition_infos, prev_transition_infos):
        ''' 
            Reward function used by HER for relabeling rewards
        
            Environment reward consists of place reward, xy-position reward
            Place reward = -|current z position - expected z position|
            Xy-position reward = -||current xy position - init xy position||^2
        '''
        batch_size = goals.shape[0]
        rewards, dones, infos = [], [], []
        
        for transition in range(batch_size):
            
            new_achieved_goal = new_achieved_goals[transition]
            goal = goals[transition]
            prev_ob = prev_obs[transition]
            new_ob = new_obs[transition]
            transition_info = transition_infos[transition]
            prev_transition_info = prev_transition_infos[transition]
            env_config = transition_info['env_config']
            sim_data_qpos = transition_info['sim_data_qpos']
            init_box_pos = transition_info['init_box_pos']
            min_height = transition_info['min_height']
            gripper_centers = transition_info['gripper_centers']
            grasp_rewards = transition_info['grasp_rewards']
            inair_rewards = transition_info['inair_rewards']
            
            done = False
            off_table = [False, False]
            info = {}
            cube_pos_coords = [None, None]
            cube_z = [None, None]
            prev_cube_pos_coords = [None, None]
            
            if env_config['train_left']:
                # cube_pos_coords[0] = new_ob['cube1'][:3]
                cube_pos_coords[0] = transition_info['cube1_pos']
                cube_z[0] = cube_pos_coords[0][-1]
                # prev_cube_pos_coords[0] = prev_ob['cube1'][:3]
                prev_cube_pos_coords[0] = prev_transition_info['cube1_pos']
                
            if env_config['train_right']:
                # cube_pos_coords[1] = new_ob['cube2'][:3]
                cube_pos_coords[1] = transition_info['cube2_pos']
                cube_z[1] = cube_pos_coords[1][-1]
                # prev_cube_pos_coords[1] = prev_ob['cube2'][:3]
                prev_cube_pos_coords[1] = prev_transition_info['cube2_pos']
                
            
            # get init box pos
            # if self._t == 0:
            #     self._init_box_pos = [new_ob['cube{}'.format(i + 1)][:3] for i in range(2)]
    
            # object placing reward before placing is finished
            in_hand = [True, True]
            target_xy_rewards = [0, 0]
            target_z_rewards = [0, 0]
            target_dist_xy = [0, 0]
            target_dist_z = [0, 0]
            prev_target_dist_xy = [0, 0]
            prev_target_dist_z = [0, 0]
            grasp_count = [0, 0]
            move_finish_rewards = [0, 0]
            stage = ['move'] * 2
            prev_stage = ['move'] * 2
            
            
            for i in range(2):
                if env_config['train_left'] and i == 1:
                    continue
                if env_config['train_right'] and i == 0:
                    continue
                dist_cube_hand = np.linalg.norm(new_ob['cube{}'.format(i + 1)][:3] - gripper_centers[i])
                in_hand[i] = dist_cube_hand < 0.08
                off_table[i] = cube_z[i] < 0.8
    
                # place reward
                cube_pos = cube_pos_coords[i]
                target_dist_xy[i] = float(np.linalg.norm(cube_pos[:2] - goal[:2]))
                target_dist_z[i] = float(np.abs(cube_pos[-1] - goal[-1]))
                
                prev_cube_pos = prev_cube_pos_coords[i]
                prev_target_dist_xy[i] = float(np.linalg.norm(prev_cube_pos[:2] - goal[:2]))
                prev_target_dist_z[i] = float(np.abs(prev_cube_pos[-1] - goal[-1]))
                
                if target_dist_xy[i] < 0.06 and cube_pos[-1] > min_height:
                    stage[i] = 'place'
                if prev_target_dist_xy[i] < 0.06 and prev_cube_pos[-1] > min_height:
                    prev_stage[i] = 'place'
                
                if prev_stage == 'move' and stage == 'place':
                    move_finish_rewards[i] += env_config['move_finish_reward']
                
                if stage[i] == 'move':
                    height_decrease = init_box_pos[i][-1] - cube_pos[-1]
                    target_z_rewards[i] = -env_config["target_z_reward"] * max(0, height_decrease)
    
                if stage[i] == 'place':
                    target_z_rewards[i] = -env_config["target_z_reward"] * target_dist_z[i]
                target_xy_rewards[i] -= env_config["target_xy_reward"] * target_dist_xy[i]
    
            # success criteria
            success = True
            if env_config['train_left']:
                success &= in_hand[0] and target_dist_xy[0] < 0.04 and target_dist_z[0] < 0.03 and stage[0] == 'place'
            if env_config['train_right']:
                success &= in_hand[1] and target_dist_xy[1] < 0.04 and target_dist_z[1] < 0.03 and stage[1] == 'place'
    
            # success reward
            success_reward = 0
            if success:
                print('All places success!')
                success_reward = env_config["success_reward"]
    
            done = success
            if env_config['train_left']:
                done |= (not in_hand[0])
            if env_config['train_right']:
                done |= (not in_hand[1])
    
            reward = success_reward 
            in_hand_rewards = [0, 0]
            if env_config["train_left"]:
                done |= off_table[0]
                reward += target_xy_rewards[0] + target_z_rewards[0] + grasp_rewards[0] + inair_rewards[0] + move_finish_rewards[0]
            if env_config["train_right"]:
                done |= off_table[1]
                reward += target_xy_rewards[1] + target_z_rewards[1] + grasp_rewards[1] + inair_rewards[1] + move_finish_rewards[1]
                
            # if env_config["train_left"]:
            #     done |= off_table[0]
            #     # reward += target_xy_rewards[0] + target_z_rewards[0] + grasp_rewards[0] + inair_rewards[0] + move_finish_rewards[0]
            #     # reward = target_xy_rewards[0]
            #     # reward = target_z_rewards[0]
            #     # reward = grasp_rewards[0]
            #     # reward = inair_rewards[0]
            #     reward = move_finish_rewards[0]
            # if env_config["train_right"]:
            #     done |= off_table[1]
            #     # reward += target_xy_rewards[1] + target_z_rewards[1] + grasp_rewards[1] + inair_rewards[1] + move_finish_rewards[1]
            #     # reward = target_xy_rewards[1]
            #     # reward = target_z_rewards[1]
            #     # reward = grasp_rewards[1]
            #     # reward = inair_rewards[1]
            #     reward = move_finish_rewards[1]
                
            # reward = goal[0]
            
            # print('her target_xy_rewards:', target_xy_rewards)
            # print('her target_z_rewards:', target_z_rewards)
            # print('her grasp_rewards:', grasp_rewards)
            # print('her inair_rewards:', inair_rewards)
            # print('her move_finish_rewards:', move_finish_rewards)
            # print('her reward:', reward)
    
            info = {"reward_xy_1": target_xy_rewards[0],
                    "reward_xy_2": target_xy_rewards[1],
                    "reward_z_1": target_z_rewards[0],
                    "reward_z_2": target_z_rewards[1],
                    "reward_grasp_1": grasp_rewards[0],
                    "reward_grasp_2": grasp_rewards[1],
                    "reward_inair_1": inair_rewards[0],
                    "reward_inair_2": inair_rewards[1],
                    "reward_move_finish_1": move_finish_rewards[0],
                    "reward_move_finish_2": move_finish_rewards[1],
                    "grasp_count_1": grasp_count[0],
                    "grasp_count_2": grasp_count[1],
                    "in_hand": in_hand,
                    "target_pos": goal,
                    "cube1_pos": cube_pos[0],
                    "cube2_pos": cube_pos[1],
                    "gripper1_pos": gripper_centers[0],
                    "gripper2_pos": gripper_centers[1],
                    "target_dist_xy": np.round(target_dist_xy, 3),
                    "target_dist_z": np.round(target_dist_z, 3),
                    "curr_qpos_l": np.round(sim_data_qpos[1:10], 1).tolist(),
                    "curr_qpos_r": np.round(sim_data_qpos[10:19], 1).tolist(),
                    "stage": stage,
                    "success": success }
            
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        R_ = np.array(rewards, dtype=np.float32)

        # return R_, np.array(dones), np.array(infos, dtype=object)
        return R_

    def _step(self, a):
        prev_reward, _, _ = self.compute_reward()

        # build action from policy output
        filled_action = np.zeros((self.action_space.size,))
        inclusion_dict = { 'right_arm': self._env_config['train_right'],
                           'left_arm': self._env_config['train_left'] }
        dest_idx, src_idx = 0, 0
        for k in self.action_space.shape.keys():
            new_dest_idx = dest_idx + self.action_space.shape[k]
            if inclusion_dict[k]:
                new_src_idx = src_idx + self.action_space.shape[k]
                filled_action[dest_idx:new_dest_idx] = a[src_idx:new_src_idx]
                src_idx = new_src_idx
            dest_idx = new_dest_idx
        a = filled_action

        # scale actions from [-1, 1] range to actual control range
        mins = self.action_space.minimum
        maxs = self.action_space.maximum
        scaled_action = np.zeros_like(a)
        for i in range(self.action_space.size):
            scaled_action[i] = mins[i] + (maxs[i] - mins[i]) * (a[i] / 2 + 0.5)

        self.do_simulation(scaled_action)
        self._t += 1

        ob = self._get_obs()

        reward, done, info = self.compute_reward()
        ctrl_reward = self._ctrl_reward(scaled_action)
        info['reward_ctrl'] = ctrl_reward

        obs_and_goals = self.add_goals(ob)
        self._reward = reward - prev_reward + ctrl_reward
        return obs_and_goals, self._reward, done, info
        # return obs_and_goals, reward, done, info
    
    def add_goals(self, ob):
        obs_and_goals = {'observation': ob}
        obs_and_goals['achieved_goal'] = np.array(self._get_pos('cube1'), dtype=np.float32) if self._env_config["train_left"] else np.array(self._get_pos('cube2'), dtype=np.float32)
        obs_and_goals['desired_goal'] = np.array(self._env_config['dest_pos'], dtype=np.float32)
        return obs_and_goals

    def reset_box(self):
        self.cube1_target_reached = False
        self.cube2_target_reached = False

        super().reset_box()

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set agent's and box's initial position from saved poses
        if self._env_config['init_qpos_dir']:
            filepath = os.path.join(self._env_config['init_qpos_dir'], 'success_qpos.p')
            with h5py.File(filepath, 'r', libver='latest', swmr=True) as f:
                select_success = False
                while not select_success:
                    ix = np.random.randint(len(f))
                    qpos = f[str(ix)].value
                    cube_l_ok = qpos[20] > self._min_height or (not self._env_config['train_left'])
                    cube_r_ok = qpos[27] > self._min_height or (not self._env_config['train_right'])
                    if cube_l_ok and cube_r_ok:
                        select_success = True

        self.set_state(qpos, qvel)

        self._hold_duration = [0, 0]

        self._t = 0
        self._placed = False
        self._init_grasp_count = [0, 0]
        self._stage = ['move'] * 2
        
    def reset(self):
        # new_goal = [np.random.uniform(-0.2,0.2), np.random.uniform(-0.2,0.2), 0.86]
        
        new_goal1 = [0., 0.1, 0.86]
        new_goal2 = [0.3, -0.02, 0.86]
        new_goal = new_goal1 if np.random.rand() > 0.5 else new_goal2
        
        self._env_config.update(
            {
                'dest_pos': new_goal
            }
        )
        goal1_index = self.sim.model.geom_name2id("goal1")
        self.sim.model.geom_pos[goal1_index] = new_goal
        self.sim.step()
        ob = super().reset()
        return self.add_goals(ob)
