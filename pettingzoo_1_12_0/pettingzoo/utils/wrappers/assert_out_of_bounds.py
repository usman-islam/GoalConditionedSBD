from gym.spaces import Discrete

from .base import BaseWrapper


class AssertOutOfBoundsWrapper(BaseWrapper):
    '''
    this wrapper crashes for out of bounds actions
    Should be used for Discrete spaces
    '''
    def __init__(self, env):
        super().__init__(env)
        assert all(isinstance(space, Discrete) for space in getattr(self, 'action_spaces', {}).values()), "should only use AssertOutOfBoundsWrapper for Discrete spaces"

    def step(self, action):
        assert (action is None and self.dones[self.agent_selection]) or self.action_space(self.agent_selection).contains(action), "action is not in action space"
        super().step(action)

    def __str__(self):
        return str(self.env)
