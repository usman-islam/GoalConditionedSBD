import os
from collections import OrderedDict

import numpy as np
import torch

from rl.sac_agent import SACAgent
from rl.normalizer import Normalizer
from util.logger import logger
from util.pytorch import to_tensor, get_ckpt_path
from env.action_spec import ActionSpec


class LowLevelAgent(SACAgent):
    """
    Low level agent that includes skill sets for each agent, their
    execution procedure given observation and skill selections from
    meta-policy, and their training (for single-skill-per-agent cases
    only).
    
    Currently the low level agent must represent fixed pretrained primitives
    and cannot be fine-tuned alongside the meta-policy training.
    """

    def __init__(self, config, ob_spaces, ac_spaces, actor, critic):
        # super().__init__(config, None, None, None, actor, critic)
        self._ob_spaces = ob_spaces
        self._ac_spaces = ac_spaces
        self._config = config

        self._build_actor(actor)
        self._log_creation()

    def _log_creation(self):
        """ Logs the structure of low-level policies. """
        if self._config.is_chef:
            logger.info('Creating a low-level agent')
            for cluster, skills in zip(self._clusters, self._subdiv_skills):
                logger.warn('Part {} has skills {}'.format(cluster, skills))
                
    def _is_goal_conditioned(self, skill):
        return 'gc' in skill
    
    def _standardize_goal(self, goal):
        xl = self._config.goal_x_lower
        xu = self._config.goal_x_upper
        yl = self._config.goal_y_lower
        yu = self._config.goal_y_upper
        zl = self._config.goal_z_lower
        zu = self._config.goal_z_upper
        
        dx = (xu - xl)/2
        dy = (yu - yl)/2
        dz = (zu - zl)/2
        
        centers = np.array([xl + dx, yl + dy, zl + dz])
        
        return centers + np.multiply(goal, np.array([dx, dy, dz]))

    def _build_actor(self, actor):
        config = self._config

        # parse body parts and skills
        if config.subdiv:
            # subdiv: 'ob1,ob2-ac1/ob3,ob4-ac2/...'
            clusters = config.subdiv.split('/')
            clusters = [
                (cluster.split('-')[0].split(','), cluster.split('-')[1].split(',')) for cluster in clusters
            ]
        else:
            clusters = [(self.ob_space.keys(), self.ac_space.shape.keys())]

        if config.subdiv_skills:
            subdiv_skills = config.subdiv_skills.split('/')
            subdiv_skills = [
                skills_list_str.split(',') for skills_list_str in subdiv_skills
            ]
        else:
            subdiv_skills = [['primitive']] * len(clusters)

        assert len(subdiv_skills) == len(clusters), \
            'subdiv_skills and clusters have different # subdivisions'

        self._clusters = clusters
        self._subdiv_skills = subdiv_skills

        self._actors = []
        self._ob_norms = []
        self._actor_gc_flags = []

        # load networks
        for cluster_idx, skills_list in enumerate(self._subdiv_skills):
            # ob_space = OrderedDict([(k, self._ob_space[k]) for k in cluster[0]])
            # if self._config.diayn:
            #     ob_space[','.join(cluster[0]) + '_diayn'] = self._config.z_dim
            # ac_decomposition = OrderedDict([(k, self._ac_space.shape[k]) for k in cluster[1]])
            # ac_size = sum(self._ac_space.shape[k] for k in cluster[1])
            # ac_space = ActionSpec(ac_size, -1, 1)
            # ac_space.decompose(ac_decomposition)

            skill_actors = []
            skill_ob_norms = []
            skill_gc_flags = []

            for skill_idx, skill in enumerate(skills_list):

                ob_space = self._ob_spaces[cluster_idx][skill_idx]
                if self._config.diayn:
                    keys_str = ','.join(list(ob_space.keys()))
                    ob_space[keys_str + '_diayn'] = self._config.z_dim
                    
                ac_space = self._ac_spaces[cluster_idx][skill_idx]
                
                skill_actor = actor(config, ob_space, ac_space, config.tanh_policy)
                skill_ob_norm = Normalizer(ob_space,
                                            default_clip_range=config.clip_range,
                                            clip_obs=config.clip_obs)

                if self._config.meta_update_target == 'HL':
                    path = os.path.join(config.subdiv_skill_dir, skill)
                    ckpt_path, ckpt_num = get_ckpt_path(path, None)
                    logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
                    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

                    if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
                        # backward compatibility to older checkpoints
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
                    else:
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0][0])
                    skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

                skill_actor.to(config.device)
                skill_actors.append(skill_actor)
                skill_ob_norms.append(skill_ob_norm)
                skill_gc_flags.append(self._is_goal_conditioned(skill))

            self._actors.append(skill_actors)
            self._ob_norms.append(skill_ob_norms)
            self._actor_gc_flags.append(skill_gc_flags)
            
            
    def state_dict(self):
        return {
            'actor_state_dict': [[_actor.state_dict() for _actor in _agent] for _agent in self._actors],
            'ob_norm_state_dict': [[_ob_norm.state_dict() for _ob_norm in _agent] for _agent in self._ob_norms],
        }

    def load_state_dict(self, ckpt):
        for _agent, agent_ckpt in zip(self._actors, ckpt['actor_state_dict']):
            for _actor, actor_ckpt in zip(_agent, agent_ckpt):
                _actor.load_state_dict(actor_ckpt)
        for _agent, agent_ckpt in zip(self._ob_norms, ckpt['ob_norm_state_dict']):
            for _ob_norm, ob_norm_ckpt in zip(_agent, agent_ckpt):
                _ob_norm.load_state_dict(ob_norm_ckpt)
        self._network_cuda(self._config.device)
                
                
    def _network_cuda(self, device):
        for _agent in self._actors:
            for _actor in _agent:
                _actor.to(device)
            
            
    # def update_normalizer(self, obs, meta_acs):
    #     """ Updates normalizers. """
    #     if self._config.ob_norm:
    #         if self._config.meta == 'hard':
                
    #             primitive_wise_obs = []
    #             for agent_ob_norms in self._ob_norms:
    #                 agent_wise_obs = []
    #                 for skill_ob_norm in agent_ob_norms:
    #                     agent_wise_obs.append([])
    #                 primitive_wise_obs.append(agent_wise_obs)
                    
    #             clusters = [','.join(item[1]) for item in self._clusters]
                        
    #             for ob, meta_ac in zip(obs, meta_acs):
    #                 for cluster_idx, cluster in enumerate(clusters):
    #                     skill_idx = meta_ac[cluster][0]
    #                     ob_ = ob.copy()
    #                     if self._actor_gc_flags[cluster_idx][skill_idx]:
    #                         ob_['goals'] = self._standardize_goal(
    #                             meta_ac[cluster + '_goals'][0]
    #                         )
    #                     if self._config.diayn:
    #                         z_name = self._actors[cluster_idx][skill_idx].z_name
    #                         ob_[z_name] = meta_ac[cluster + '_diayn']
    #                     primitive_wise_obs[cluster_idx][skill_idx].append(ob_)
                                            
    #             for cluster_idx, agent_ob_norms in enumerate(self._ob_norms):
    #                 for skill_idx, skill_ob_norm in enumerate(agent_ob_norms):
    #                     skill_ob_norm.update(primitive_wise_obs[cluster_idx][skill_idx])
    #                     skill_ob_norm.recompute_stats()
    
    
    # def update_normalizer(self, obs, meta_acs):
    #     """ Updates normalizers. """
    #     if self._config.ob_norm:
    #         if self._config.meta == 'hard':
                    
    #             clusters = [','.join(item[1]) for item in self._clusters]
                        
    #             for ob, meta_ac in zip(obs, meta_acs):
    #                 for cluster_idx, cluster in enumerate(clusters):
    #                     skill_idx = meta_ac[cluster][0]
    #                     ob_ = ob.copy()
    #                     if self._config.plan_goals and self._actor_gc_flags[cluster_idx][skill_idx]:
    #                         ob_['goals'] = self._standardize_goal(
    #                             meta_ac[cluster + '_goals']
    #                         )
    #                     if self._config.diayn:
    #                         z_name = self._actors[cluster_idx][skill_idx].z_name
    #                         ob_[z_name] = meta_ac[cluster + '_diayn']
    #                     self._ob_norms[cluster_idx][skill_idx].update(ob_)
    #                     self._ob_norms[cluster_idx][skill_idx].recompute_stats()
    
    
    def update_normalizer(self, obs):
        pass
            

    def act(self, ob, meta_ac, is_train=True):
        """
        Returns action and the actor's activation given an observation @ob and meta action @meta_ac for rollout.
        """
        ac = OrderedDict()
        activation = OrderedDict()
        if self._config.meta == 'hard':
            for i, agent_ac in enumerate(meta_ac.values()):
                if list(meta_ac.keys())[i].endswith('_diayn') or list(meta_ac.keys())[i].endswith('_goals'):
                    # skip diayn and goal outputs from meta-policy
                    continue
                skill_idx = agent_ac[0]
                ob_ = ob.copy()
                if self._config.plan_goals and self._actor_gc_flags[i][skill_idx]:
                    ob_['goals'] = self._standardize_goal(
                        meta_ac[list(meta_ac.keys())[i] + '_goals']
                    )
                if self._config.diayn:
                    z_name = self._actors[i][skill_idx].z_name
                    ob_[z_name] = meta_ac[list(meta_ac.keys())[i] + '_diayn']
                ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
                ob_ = to_tensor(ob_, self._config.device)
                if self._config.meta_update_target == 'HL':
                    ac_, activation_ = self._actors[i][skill_idx].act(ob_, False)
                else:
                    ac_, activation_ = self._actors[i][skill_idx].act(ob_, is_train)
                ac.update(ac_)
                activation.update(activation_) 
        return ac, activation

    def act_log(self, ob, meta_ac=None):
        """
        Returns action and the actor's activation given an observation @ob and meta action @meta_ac for updating networks.
        Note: only usable for SAC agents.
        """
        ob_detached = { k: v.detach().cpu().numpy() for k, v in ob.items() }

        ac = OrderedDict()
        log_probs = []
        meta_ac_keys = [k for k in meta_ac.keys() if not (k.endswith('_diayn') or k.endswith('_goals'))]
        for i, key in enumerate(meta_ac_keys):
            #skill_idx = meta_ac[key]
            skill_idx = 0

            ob_ = ob_detached.copy()
            if self._config.diayn:
                z_name = self._actors[i][skill_idx].z_name
                # ob_[z_name] = meta_ac[z_name].detach().cpu().numpy()
                ob_[z_name] = meta_ac[key  + '_diayn'].detach().cpu().numpy()
            ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
            ob_ = to_tensor(ob_, self._config.device)
            ac_, log_probs_ = self._actors[i][skill_idx].act_log(ob_)
            ac.update(ac_)
            log_probs.append(log_probs_)

        try:
            log_probs = torch.cat(log_probs, -1).sum(-1, keepdim=True)
        except Exception:
            import pdb; pdb.set_trace()

        return ac, log_probs

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
            self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass



# # Original code

# import os
# from collections import OrderedDict

# import numpy as np
# import torch

# from rl.sac_agent import SACAgent
# from rl.normalizer import Normalizer
# from util.logger import logger
# from util.pytorch import to_tensor, get_ckpt_path
# from env.action_spec import ActionSpec


# class LowLevelAgent(SACAgent):
#     """
#     Low level agent that includes skill sets for each agent, their
#     execution procedure given observation and skill selections from
#     meta-policy, and their training (for single-skill-per-agent cases
#     only).
#     """

#     def __init__(self, config, ob_space, ac_space, actor, critic):
#         super().__init__(config, ob_space, ac_space, actor, critic)

#     def _log_creation(self):
#         """ Logs the structure of low-level policies. """
#         if self._config.is_chef:
#             logger.info('Creating a low-level agent')
#             for cluster, skills in zip(self._clusters, self._subdiv_skills):
#                 logger.warn('Part {} has skills {}'.format(cluster, skills))

#     def _build_actor(self, actor):
#         config = self._config

#         # parse body parts and skills
#         if config.subdiv:
#             # subdiv: 'ob1,ob2-ac1/ob3,ob4-ac2/...'
#             clusters = config.subdiv.split('/')
#             clusters = [
#                 (cluster.split('-')[0].split(','), cluster.split('-')[1].split(',')) for cluster in clusters
#             ]
#         else:
#             clusters = [(ob_space.keys(), ac_space.shape.keys())]

#         if config.subdiv_skills:
#             subdiv_skills = config.subdiv_skills.split('/')
#             subdiv_skills = [
#                 skills.split(',') for skills in subdiv_skills
#             ]
#         else:
#             subdiv_skills = [['primitive']] * len(clusters)

#         assert len(subdiv_skills) == len(clusters), \
#             'subdiv_skills and clusters have different # subdivisions'

#         self._clusters = clusters
#         self._subdiv_skills = subdiv_skills

#         self._actors = []
#         self._ob_norms = []

#         # load networks
#         for cluster, skills in zip(self._clusters, self._subdiv_skills):
#             ob_space = OrderedDict([(k, self._ob_space[k]) for k in cluster[0]])
#             if self._config.diayn:
#                 ob_space[','.join(cluster[0]) + '_diayn'] = self._config.z_dim
#             ac_decomposition = OrderedDict([(k, self._ac_space.shape[k]) for k in cluster[1]])
#             ac_size = sum(self._ac_space.shape[k] for k in cluster[1])
#             ac_space = ActionSpec(ac_size, -1, 1)
#             ac_space.decompose(ac_decomposition)

#             skill_actors = []
#             skill_ob_norms = []
#             for skill in skills:
#                 skill_actor = actor(config, ob_space, ac_space, config.tanh_policy)
#                 skill_ob_norm = Normalizer(ob_space,
#                                             default_clip_range=config.clip_range,
#                                             clip_obs=config.clip_obs)

#                 if self._config.meta_update_target == 'HL':
#                     path = os.path.join(config.subdiv_skill_dir, skill)
#                     ckpt_path, ckpt_num = get_ckpt_path(path, None)
#                     logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
#                     ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

#                     if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
#                         # backward compatibility to older checkpoints
#                         skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
#                     else:
#                         skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0][0])
#                     skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

#                 skill_actor.to(config.device)
#                 skill_actors.append(skill_actor)
#                 skill_ob_norms.append(skill_ob_norm)

#             self._actors.append(skill_actors)
#             self._ob_norms.append(skill_ob_norms)

#     def act(self, ob, meta_ac, is_train=True):
#         """
#         Returns action and the actor's activation given an observation @ob and meta action @meta_ac for rollout.
#         """
#         ac = OrderedDict()
#         activation = OrderedDict()
#         if self._config.meta == 'hard':
#             for i, skill_idx in enumerate(meta_ac.values()):
#                 if [k for k in meta_ac.keys()][i].endswith('_diayn'):
#                     # skip diayn outputs from meta-policy
#                     continue
#                 skill_idx = skill_idx[0]
#                 ob_ = ob.copy()
#                 if self._config.diayn:
#                     z_name = self._actors[i][skill_idx].z_name
#                     ob_[z_name] = meta_ac[self._actors[i][skill_idx].z_name]
#                 ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
#                 ob_ = to_tensor(ob_, self._config.device)
#                 if self._config.meta_update_target == 'HL':
#                     ac_, activation_ = self._actors[i][skill_idx].act(ob_, False)
#                 else:
#                     ac_, activation_ = self._actors[i][skill_idx].act(ob_, is_train)
#                 ac.update(ac_)
#                 activation.update(activation_)

#         return ac, activation

#     def act_log(self, ob, meta_ac=None):
#         """
#         Returns action and the actor's activation given an observation @ob and meta action @meta_ac for updating networks.
#         Note: only usable for SAC agents.
#         """
#         ob_detached = { k: v.detach().cpu().numpy() for k, v in ob.items() }

#         ac = OrderedDict()
#         log_probs = []
#         meta_ac_keys = [k for k in meta_ac.keys() if (not k.endswith('_diayn'))]
#         for i, key in enumerate(meta_ac_keys):
#             #skill_idx = meta_ac[key]
#             skill_idx = 0

#             ob_ = ob_detached.copy()
#             if self._config.diayn:
#                 z_name = self._actors[i][skill_idx].z_name
#                 ob_[z_name] = meta_ac[z_name].detach().cpu().numpy()
#             ob_ = self._ob_norms[i][skill_idx].normalize(ob_)
#             ob_ = to_tensor(ob_, self._config.device)
#             ac_, log_probs_ = self._actors[i][skill_idx].act_log(ob_)
#             ac.update(ac_)
#             log_probs.append(log_probs_)

#         try:
#             log_probs = torch.cat(log_probs, -1).sum(-1, keepdim=True)
#         except Exception:
#             import pdb; pdb.set_trace()

#         return ac, log_probs

#     def sync_networks(self):
#         if self._config.meta_update_target == 'LL' or \
#             self._config.meta_update_target == 'both':
#             super().sync_networks()
#         else:
#             pass


