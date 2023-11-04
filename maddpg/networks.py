import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
"""
Notice:
        ActorNetwork 部分 网络输出需要根据不同的任务进行修改
Netwroks Args:
    CriticNetwork:
        beta: CriticNetwork learning rate
        critic_state_dims: critic 网络输入中 state 部分
        fc1_out_dims: net 1 out_dims
        fc2_out_dims: net 2 out_dims
        n_agents:
        n_actions: list 包含所有的动作空间
        name: eg: agent_1_actor.pth
        chkpt_dir: eg: tmp/maddpg/
    
    ActorNetwork:
        alpha: ActorNetwork learning rate
        actor_state_dims: 对于单个智能体的状态输入 同理 也是根据一个 list 进行抽值
        n_actions: list 包含所有的动作空间
        n_actions_single: number 当前智能体 Agent 的动作空间 size
        fc1_out_dims: net 1 out_dims
        fc2_out_dims: net 2 out_dims
        name: eg: agent_1_actor.pth
        chkpt_dir: eg: tmp/maddpg/ save_module/maddpg/
"""
class CriticNetwork(nn.Module):
    def __init__(self, beta, critic_state_dims, fc1_out_dims, fc2_out_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.beta = beta
        self.critic_state_dims = critic_state_dims
        self.fc1_out_dims = fc1_out_dims
        self.fc2_out_dims = fc2_out_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir

        # n_actions 定义为 list 虽然可能环境中的 action_dims 是一样的
        # CriticNetwork_input_dims 是 critic_states_dims + critic_actions_dims
        self.critic_actions_dims = sum(self.n_actions)

        self.chkpt_file = os.path.join(self.chkpt_dir, self.name)

        self.fc1 = nn.Linear(self.critic_state_dims + self.critic_actions_dims, self.fc1_out_dims)
        self.fc2 = nn.Linear(self.fc1_out_dims, self.fc2_out_dims)
        self.q = nn.Linear(self.fc2_out_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.beta)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, actor_state_dims, fc1_out_dims, fc2_out_dims, 
                 n_actions, n_actions_single, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.alpha = alpha
        self.actor_state_dims = actor_state_dims
        self.fc1_out_dims = fc1_out_dims
        self.fc2_out_dims = fc2_out_dims
        self.n_actions = n_actions
        self.n_actions_single = n_actions_single
        self.name = name
        self.chkpt_dir = chkpt_dir

        self.chkpt_file = os.path.join(chkpt_dir, self.name)

        self.fc1 = nn.Linear(self.actor_state_dims, self.fc1_out_dims)
        self.fc2 = nn.Linear(self.fc1_out_dims, self.fc2_out_dims)
        self.pi = nn.Linear(self.fc2_out_dims, self.n_actions_single)

        self.optimizer = optim.Adam(self.parameters(), lr = self.alpha)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, actor_data):
        x = self.fc1(actor_data)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.pi(x)
        pi = torch.softmax(x, dim=1)

        return pi

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))