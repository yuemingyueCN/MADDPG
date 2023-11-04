import torch
from .networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, alpha, actor_state_dims, actor_fc1, actor_fc2, n_actions_single, 
                 beta, critic_state_dims, critic_fc1, critic_fc2, 
                 n_agents, n_actions, agent_idx, chkpt_dir, gamma, tau):
        
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_actions_single = n_actions_single
        self.agent_idx = agent_idx
        self.chkpt_dir = chkpt_dir

        self.agent_name = 'agent_{}'.format(self.agent_idx)

        self.actor= ActorNetwork(alpha=alpha, actor_state_dims=actor_state_dims,
                                 fc1_out_dims=actor_fc1, fc2_out_dims=actor_fc2,
                                 n_actions=self.n_actions, n_actions_single=self.n_actions_single,
                                 name=self.agent_name+'_actor.pth', chkpt_dir=self.chkpt_dir)
        self.critic = CriticNetwork(beta=beta, critic_state_dims=critic_state_dims,
                                    fc1_out_dims=critic_fc1, fc2_out_dims=critic_fc2,
                                    n_agents=self.n_agents, n_actions=self.n_actions,
                                    name=self.agent_name+'_critic.pth', chkpt_dir=self.chkpt_dir)
        self.target_actor = ActorNetwork(alpha=alpha, actor_state_dims=actor_state_dims,
                                 fc1_out_dims=actor_fc1, fc2_out_dims=actor_fc2,
                                 n_actions=self.n_actions, n_actions_single=self.n_actions_single,
                                 name=self.agent_name+'_target_actor.pth', chkpt_dir=self.chkpt_dir)
        self.target_critic = CriticNetwork(beta=beta, critic_state_dims=critic_state_dims,
                                    fc1_out_dims=critic_fc1, fc2_out_dims=critic_fc2,
                                    n_agents=self.n_agents, n_actions=self.n_actions,
                                    name=self.agent_name+'_target_critic.pth', chkpt_dir=self.chkpt_dir)

        self.update_network_parameters(tau_=1)

    # 这一步进行 soft update
    def update_network_parameters(self, tau_=None):
        if tau_ is None:
            tau_ = self.tau
        
        # 对 target_actor soft update
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau_*actor_state_dict[name].clone() + \
                    (1-tau_)*target_actor_state_dict[name].clone()
            
        self.target_actor.load_state_dict(actor_state_dict)

        # 对 target_critic soft update
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau_*critic_state_dict[name].clone() + \
                    (1-tau_)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, obs):
        state = torch.tensor([obs], dtype=torch.float).to(self.actor.device)
        actions = self.actor.forward(state)
        # 噪声 : noise
        # noise = torch.rand(self.n_actions).to(self.actor.device)
        # actions = actions + noise
        return actions.detach().cpu().to(self.actor.device)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
