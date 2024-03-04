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

    def learn(self, writer, step):
        if not self.buffer.ready():
            return

        # 这里增加一个高斯噪声判定迭代器 每迭代一次 +1
        self.noise_count = self.noise_count + 1
        if self.noise_count >= self.noise_max:
            self.noise = False

        # 把数据从经验池中 buffer 出来
        critic_states, actor_states, actions, rewards, \
        critic_states_next, actor_states_next, terminal = self.buffer.sample_buffer()

        # 转成 torch tensor
        critic_states = torch.tensor(critic_states, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        critic_states_next = torch.tensor(critic_states_next, dtype=torch.float).to(self.device)
        terminal = torch.tensor(terminal).to(self.device)

        for idx in range(self.n_agents):
            actor_states[idx] = torch.tensor(actor_states[idx], dtype=torch.float).to(self.device)
            actions[idx] = torch.tensor(actions[idx], dtype=torch.float).to(self.device)
            actor_states_next[idx] = torch.tensor(actor_states_next[idx], dtype=torch.float).to(self.device)

        actions_next = []
        with torch.no_grad():
            for agent_idx, agent in enumerate(self.agents):
                # 根据 s_ 和 target_network 求得 a_
                new_pi = agent.target_actor.forward(actor_states_next[agent_idx])
                actions_next.append(new_pi)
                
        # 用于 critic 网络的动作 tensor
        new_actions_tensor = torch.cat([acts for acts in actions_next], dim=1)
        old_actions_tensor = torch.cat([acts for acts in actions], dim=1)

        # 循环梯度计算更新
        # 遍历每一个智能体 agent
        for agent_idx, agent in enumerate(self.agents):
            """
            更新 critic
            """
            # 计算 target_Q
            with torch.no_grad():
                critic_value_ = agent.target_critic.forward(critic_states_next, new_actions_tensor).flatten()
                critic_value_[terminal[:, agent_idx]] = 0.0
                target_Q = rewards[:, agent_idx] + agent.gamma * critic_value_
            # 计算 current_Q
            current_Q = agent.critic.forward(critic_states, old_actions_tensor).flatten()
            critic_loss = F.mse_loss(target_Q, current_Q)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_critic_loss', critic_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_critic_loss', critic_loss.item(), step)

            """
            更新 actor
            """
            # 重新选择动作 其余智能体动作不变
            tmp_actions = [t.clone().detach() for t in actions]
            tmp_actions[agent_idx] = agent.actor(actor_states[agent_idx])
            ten_actions_tensor = torch.cat([acts for acts in tmp_actions], dim=1)
            actor_loss = agent.critic.forward(critic_states, ten_actions_tensor).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()

            if agent.agent_name == "agent_0":
                writer.add_scalar('loss/agent_0_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_1":
                writer.add_scalar('loss/agent_1_actor_loss', actor_loss.item(), step)
            if agent.agent_name == "agent_2":
                writer.add_scalar('loss/agent_2_actor_loss', actor_loss.item(), step)

        # 最后统一对 target_networks 进行软更新
        for idx, agent in enumerate(self.agents):
            agent.update_network_parameters()
