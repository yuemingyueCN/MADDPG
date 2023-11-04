import torch
import torch.nn.functional as F
from .agent import Agent
from .replay_buffer import MultiAgentReplayBuffer
"""
Notice:
    这里写的MADDPG版本
    设定的每个智能体的Actor和Critic的中间层的结构是一样的
    注意自行根据要求修改
"""
class MADDPG:
    def __init__(self, 
                 alpha, actor_states_dims, actor_fc1, actor_fc2, 
                 beta, critic_fc1, critic_fc2,
                 n_actions, n_agents, chkpt_dir, gamma, tau,
                 buffer_max_size,
                 buffer_batch_size
                 ):
        # 初始化 device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 存储 actor critic 的梯度 用于统一的梯度更新
        self.gradient_list = []
        # 超参数
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.actor_states_dims = actor_states_dims
        self.critic_state_dims = sum(self.actor_states_dims)

        self.n_agents = n_agents
        self.n_actions = n_actions

        # 传递经验池参数 注意和传递的参数大小一致 分析的过程中注意 矩阵数据的维度变化
        self.buffer_max_size = buffer_max_size
        self.buffer_critic_state_dims = self.critic_state_dims
        self.buffer_actor_state_dims = self.actor_states_dims
        self.buffer_n_actions = self.n_actions
        self.buffer_n_agents = self.n_agents
        self.buffer_batch_size = buffer_batch_size
        # 定义经验池
        self.buffer = MultiAgentReplayBuffer(
            max_size = self.buffer_max_size,
            critic_state_dims = self.buffer_critic_state_dims,
            actor_state_dims = self.buffer_actor_state_dims,
            n_actions = self.buffer_n_actions,
            n_agents = self.buffer_n_agents,
            batch_size = self.buffer_batch_size
        )
        # 初始化 agents
        self.agents = []
        for idx in range(self.n_agents):
            self.agents.append(
                Agent(alpha = self.alpha,
                      actor_state_dims = self.actor_states_dims[idx],
                      actor_fc1 = actor_fc1,
                      actor_fc2 = actor_fc2,
                      n_agents = self.n_agents,
                      n_actions = self.n_actions,
                      n_actions_single = self.n_actions[idx],
                      agent_idx = idx,
                      chkpt_dir = chkpt_dir,
                      gamma = self.gamma,
                      tau = self.tau,
                      beta = self.beta,
                      critic_state_dims = self.critic_state_dims,
                      critic_fc1 = critic_fc1,
                      critic_fc2 = critic_fc2
                      )
            )

    def save_checkpoint(self):
        print('... saving checkpoint models ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint models ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, actor_state_all):
        # 接受数据是 所有 Agents 的 state 二维矩阵
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(actor_state_all[agent_idx])
            actions.append(action)
        return actions

    def learn(self):
        if not self.buffer.ready():
            return

        critic_states, actor_states, actions, rewards, \
        critic_states_next, actor_states_next, terminal = self.buffer.sample_buffer()

        critic_states = torch.tensor(critic_states, dtype=torch.float).to(self.device)

        actions_list = []
        for idx in range(self.n_agents):
            actions_list.append(torch.tensor(actions[idx], dtype=torch.float).to(self.device))

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        critic_states_next = torch.tensor(critic_states_next, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal).to(self.device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_states_next[agent_idx], dtype=torch.float).to(self.device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(self.device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions_list[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions],dim=1)
        
        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(critic_states_next, new_actions).flatten()
            critic_value_[terminal[:, agent_idx]] = 0.0
            critic_value = agent.critic.forward(critic_states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            # 保存模型的梯度
            gradient_dict = {}
            for name, param in agent.critic.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # 保存梯度到 list
            self.gradient_list.append(gradient_dict)

            actor_loss = agent.critic.forward(critic_states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)

            # 保存模型的梯度
            gradient_dict = {}
            for name, param in agent.actor.named_parameters():
                if param.grad is not None:
                    gradient_dict[name] = param.grad.clone()
            # 保存梯度到 list
            self.gradient_list.append(gradient_dict)

        for agent_idx, agent in enumerate(self.agents):
            # 为模型的参数设置梯度
            for name, param in agent.critic.named_parameters():
                if name in self.gradient_list[agent_idx * 2]:
                    param.grad = self.gradient_list[agent_idx * 2][name]

            # 为模型的参数设置梯度
            for name, param in agent.actor.named_parameters():
                if name in self.gradient_list[agent_idx * 2 + 1]:
                    param.grad = self.gradient_list[agent_idx * 2 + 1][name]

            # 更新模型的参数
            agent.critic.optimizer.step()
            agent.actor.optimizer.step()
        """
        把 self.gradient_list 归零
        否则加载的梯度一直是初始的梯度值
        self.gradient_list 会一直累加
        """
        # print(len(self.gradient_list))
        self.gradient_list = []

        # 最后统一对 target_networks 进行软更新
        for idx, agent in enumerate(self.agents):
            agent.update_network_parameters()
