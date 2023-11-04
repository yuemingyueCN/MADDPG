import argparse
"""
Notice:
    自行补充参数数据
"""
def parse_args_uav():
    parser = argparse.ArgumentParser("UAV env Hyper Parameters")
    parser.add_argument("--num_red", type=int, default=10, help="numb of red uavs")
    parser.add_argument("--num_blue", type=int, default=10, help="numb of blue uavs")
    parser.add_argument("--test", type=str, default="this is a test", help="test")

    return parser.parse_args()

def parse_args_maddpg():
    parser = argparse.ArgumentParser("MADDGP Framworks Hyper Parasmeters")
    parser.add_argument("--alpha", type=float, default=None, help="ActorNetwork learning rate")
    parser.add_argument("--beta", type=float, default=None, help="CriticNetwork learning rate")
    parser.add_argument("--actor_states_dims", type=list, default=None, help="所有agents的输入ActorNetwork的维度 eg:[3, 3] 其中有2个agent的获取动作的状态信息维度分别为 3 3")
    parser.add_argument("--actor_fc1", type=int, default=None, help="ActorNetwork linear 1 output dims")
    parser.add_argument("--actor_fc2", type=int, default=None, help="ActorNetwork linear 2 output dims")
    parser.add_argument("--critic_fc1", type=int, default=None, help="CriticNetwork linear 1 output dims")
    parser.add_argument("--critic_fc2", type=int, default=None, help="CriticNetwork linear 2 output dims")
    parser.add_argument("--n_actions", type=int, default=None, help="所有agents的动作空间维度 eg:[2,3] 有两个agent动作空间为 2 3")
    parser.add_argument("--n_agents",type=int, default=None, help="number of agents")
    parser.add_argument("--chkpt_dir", type=str, default=None, help="model save/load chkpt_dir eg':model/maddpg/'")
    parser.add_argument("--gamma", type=float, default=None, help="attenuation factor gamma")
    parser.add_argument("--tau", type=float, default=None, help="soft update parameters")
    parser.add_argument("--buffer_max_size", type=int, default=None, help="经验池最大数据容量")
    parser.add_argument("--buffer_batch_size", type=int , default=None, help="maddpg learn batch_size")

    return parser.parse_args()


