from .mlp_actor_critic import MlpActor, MlpCritic
from mrn.src.model import CriticAsymMaxSAG


def get_actor_critic_by_name(name):
    if name == 'mlp':
        return MlpActor, MlpCritic
    elif name == 'mrn':
        return MlpActor, CriticAsymMaxSAG
    else:
        raise NotImplementedError()

