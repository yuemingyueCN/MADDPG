import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_state_dims, actor_state_dims, 
                 n_actions, n_agents, batch_size):
        """
        Args:
            max_size: 最大经验池容量
            critic_state_dims: critic 部分维度 state 部分 critic_state_dims = sum(actor_state_dims)
            actor_state_dims: 数据格式 list n_agents 个数据 eg: [6, 6, 6] 指三个智能体的状态空间为 6 6 6
            n_actions: 动作空间维度 数据格式 list n_agents 个数据 eg: [3, 3, 3] 指三个智能体的动作空间为 3 3 3
            n_agents: 智能体的个数
            batch_size: batch_size
        """
        self.mem_size = max_size
        self.critic_state_dims = critic_state_dims
        self.actor_state_dims = actor_state_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size

        # 定义计数器
        self.mem_cntr = 0

        # 定义初始化 Critic 经验池
        self.critic_state_memory = np.zeros((self.mem_size, self.critic_state_dims))
        self.critic_new_state_memory = np.zeros((self.mem_size, self.critic_state_dims))

        # 定义 reward terminal
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)

        # 初始化 Actor 存储经验池
        """
        Actor 经验池逻辑
            MADDPG 算法 即是 每个 Actor 的网络接受的是自我智能体 obs
            所以为了做到兼容 初始化 Actor 的经验池 为 n_agents * self.mem_size * self.actor_state_dims[i]
        """
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_state_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_state_dims[i])))
            self.action_memory.append(
                            np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, critic_state, actor_state, action, reward, 
                            critic_state_next, actor_state_next,  terminal):
        # 定义 index 存储数据逻辑
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = actor_state[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = actor_state_next[agent_idx]
            self.action_memory[agent_idx][index] = action[agent_idx]

        self.critic_state_memory[index] = critic_state
        self.critic_new_state_memory[index] = critic_state_next
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        # 计数逻辑变量 + 1
        self.mem_cntr += 1

    def sample_buffer(self):
        # notice： buffer 出来的都是 batchz_size 维度数据
        # 逻辑控制变量
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        critic_states = self.critic_state_memory[batch]
        rewards = self.reward_memory[batch]
        critic_states_next = self.critic_new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # 按照 n_agents 索取数据
        actor_states = []
        actor_states_next = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_states_next.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.action_memory[agent_idx][batch])

        return (critic_states, actor_states, actions, rewards, 
                critic_states_next, actor_states_next, terminal)

    def ready(self):
        # 训练开始标识符
        if self.mem_cntr >= self.batch_size:
            return True