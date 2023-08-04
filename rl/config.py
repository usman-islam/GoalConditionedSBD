import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        "Skill Coordination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # environment
    parser.add_argument("--env", type=str, default="ant-push-v0",
                        help="Environment name")
    parser.add_argument("--env_args", type=str, default=None)
    parser.add_argument("--init_qpos_dir", type=str, default=None,
                        help="A list of qpos for initialization")

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "mrn"])
    parser.add_argument("--policy", type=str, default="mlp",
                        choices=["mlp"])
    parser.add_argument("--meta", type=str, default=None,
                        choices=[None, "hard"],
                        help="No meta policy is used for None"
                             "The meta policy selects one primitive skill for 'hard'"
                        )
    parser.add_argument("--meta_update_target", type=str, default="HL",
                        choices=["HL", "LL", "both"],
                        help="'HL' updates only the meta policy"
                             "'LL' updates only the low-level policy"
                             "'both' updates all networks jointly"
                        )

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=64)
    parser.add_argument("--activation", type=str, default="ReLU",
                        choices=["ReLU", "ELU", "Tanh"])
    parser.add_argument("--tanh_policy", type=str2bool, default=True)

    # coordination
    parser.add_argument("--max_meta_len", type=int, default=1)
    parser.add_argument("--fix_embedding", type=str2bool, default=False,
                        help="Fix skill embedding if meta_ac does not change")

    # observation normalization
    parser.add_argument("--ob_norm", type=str2bool, default=True)
    parser.add_argument("--max_ob_norm_step", type=int, default=int(1e7))
    parser.add_argument("--clip_obs", type=float, default=200,
                        help="the clip range of observation")
    parser.add_argument("--clip_range", type=float, default=5,
                        help="the clip range after normalization of observation")

    # off-policy rl
    # parser.add_argument("--buffer_size", type=int, default=int(1e3),
                        # help="the size of the buffer (# episodes)")
    parser.add_argument("--discount_factor", type=float, default=0.99,
                        help="the discount factor")
    parser.add_argument("--lr_actor", type=float, default=3e-4,
                        help="the learning rate of the actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4,
                        help="the learning rate of the critic")
    # parser.add_argument("--polyak", type=float, default=0.995,
    #                     help="the average coefficient")

    # multi-agent and agent-specific skills
    parser.add_argument("--subdiv", type=str, default=None,
                        help="Subdivision of observation and action space")
    parser.add_argument("--subdiv_skills", type=str, default=None,
                        help="List of primitive skills for each agent")
    parser.add_argument("--subdiv_skill_dir", type=str, default=None,
                        help="Path to the primitive skill checkpoints")

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--num_batches", type=int, default=50,
                        help="the times to update the network per epoch")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="the sample batch size")
    parser.add_argument("--max_grad_norm", type=float, default=100)
    parser.add_argument("--max_global_step", type=int, default=int(2e6))
    parser.add_argument("--gpu", type=int, default=None)

    # sac
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")

    # ppo
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)
    parser.add_argument("--rollout_length", type=int, default=1000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # skill behavior diversification (DIAYN)
    parser.add_argument("--diayn", type=str2bool, default=True)
    parser.add_argument("--z_dim", type=int, default=5)
    parser.add_argument("--z_dist", type=str, default="normal",
                        choices=["normal"])
    parser.add_argument("--discriminator_loss_weight", type=float, default=1,
                        help="the weight of discriminator policy loss")

    # log
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--evaluate_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help="set it True if you want to use wandb")

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--save_rollout", type=str2bool, default=False,
                        help="save rollout information during evaluation")
    parser.add_argument("--record", type=str2bool, default=True)
    parser.add_argument("--record_caption", type=str2bool, default=True)
    parser.add_argument("--num_record_samples", type=int, default=1,
                        help="number of trajectories to collect during eval")
    parser.add_argument("--save_qpos", type=str2bool, default=False,
                        help="save entire qpos history of successful rollouts to file (for idle primitive training)")
    parser.add_argument("--save_success_qpos", type=str2bool, default=True,
                        help="save later segment of successful rollouts to file (for placing primitive training)")

    # misc
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--notes", type=str, default="")
    # parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--virtual_display", type=str, default=":1",
                        help="Specify virtual display for rendering if you use (e.g. ':0' or ':1')")
    
    
    
    
    
    
    
    
    
    # MRN
    

    parser.add_argument('--agent', type=str, default='ddpg', choices=[
        'ddpg', 'her', 'gcsl', 'wgcsl', 'mher'
    ], help='the agent name')
    parser.add_argument('--critic', type=str, default='asym-max', choices=[
        'monolithic', 'bilinear', 'l2', 'asym', 'dn', 'wn', 'asym-max',
        'asym-lse', 'max', 'sym', 'wn-softmax', 'wn-max', 'wn-maxavg',
        'dn-softmax', 'dn-max', 'softmax', 'asym-max-sag',
        'asym-max-sag-latent', 'pqe', 'asym-new',
    ], help='the critic type')

    parser.add_argument('--n-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--n-ensembles', type=int, default=3, help='number of ensembles in dynamics model')
    parser.add_argument('--n-dynamics-updates', type=int, default=2, help='number of ensembles in dynamics model')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--relabel-rate', type=int, default=0.8, help='ratio to be replace')
    parser.add_argument('--dim-hidden', type=int, default=256, help='hidden dimension of neural networks')
    parser.add_argument('--dim-model-hidden', type=int, default=176, help='hidden dimension of neural networks')
    parser.add_argument('--dim-critic-hidden', type=int, default=176, help='hidden dimension of critic networks')
    parser.add_argument('--dim-new-hidden', type=int, default=174, help='hidden dimension of critic networks')
    parser.add_argument('--dim-embed', type=int, default=16, help='hidden dimension of embeddings')

    parser.add_argument('--dynamics-coef', type=float, default=0.0001, help='dynamics coefficient')
    parser.add_argument('--loss-scale', type=float, default=20.0, help='loss scale')

    parser.add_argument('--negative-reward', action='store_true', help='if reward is {0, 1} or {-1, 0}')
    parser.add_argument('--terminate', action='store_true', help='whether terminate at goal')

    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='./results/', help='the path to save the models')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=1024, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor') # 0.001
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic') # 0.001
    parser.add_argument('--lr-transition', type=float, default=0.001, help='the learning rate of the transition model')
    parser.add_argument('--polyak', type=float, default=0.9, help='the average coefficient')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--rollout-n-episodes', type=int, default=10, help='the rollouts per mpi') # 2
    parser.add_argument('--n-init-episodes', type=int, default=256, help='number of initial random episodes')
    parser.add_argument('--eval-rollout-n-episodes', type=int, default=100, help='the number of tests')

    parser.add_argument('--wgcsl-adv-clip', type=float, default=10, help='wgcsl clip value')
    parser.add_argument('--wgcsl-baw-delta', type=float, default=0.15, help='wgcsl clip value')
    parser.add_argument('--wgcsl-baw-max', type=float, default=80, help='wgcsl clip value')
    
    parser.add_argument('--goal_conditioned', type=bool, default=True, help='Whether to use goal conditioning')
    

    args, unparsed = parser.parse_known_args()
    args.env_args_str = args.env_args

    return args, unparsed
