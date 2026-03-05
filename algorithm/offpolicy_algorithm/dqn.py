from .baseoffpolicy import OffPolicyAlgorithm
from memory.memory import ReplayBuffer
import torch 
from utils.result import Result
from logger.logger import Logger
import numpy as np
from agent.atrai_agent import AgentBase, AtariDQNAgent


OPTIMIZER_DICT = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW,
}

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}


class DQN(OffPolicyAlgorithm):
    
    """DQN algorithm implementation."""

    def __init__(self, training_envs, testing_envs, buffer: ReplayBuffer, agent:AgentBase, logger: Logger, device, save_pth: str, args, target_agent=None):
        super(DQN, self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, args)
        
        algo_args = args.algorithm
        assert algo_args.name=="DQN", "The method name in args must be 'dqn' for DQN algorithm."
        self.lr = algo_args.learning_rate
        self.gamma = algo_args.gamma 
        self.start_epsilon = algo_args.start_epsilon
        self.end_epsilon = algo_args.end_epsilon
        self.epsilon_schedular = algo_args.epsilon_schedular
        self.use_target = algo_args.use_target 
        self.batch_size = algo_args.batch_size
        self.device = device
        
        if self.use_target:
            self.target_agent = target_agent
            self._target_hard_update()

        self.target_update_method = algo_args.target_update_method
        self.target_update_interval = algo_args.target_update_interval
        self.tau = algo_args.target_update_tau


        
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.parameters(), lr=self.lr)
        
        
    
    def _target_hard_update(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
    
    def _target_soft_update(self):
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


    def _update_buffer(self, batch):
        self.buffer.add(batch)


    def _update_policy(self):
        self.agent.train()
        with Result("train") as result:
            batch = self.buffer.sample(self.batch_size)
            states, actions, next_states, rewards, dones = batch['states'], batch['actions'], batch['next_states'], batch['rewards'], batch['dones']

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).long().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            dones = torch.from_numpy(dones).float().to(self.device)

            # calculate the DQN loss    
            if self.use_target:
                target = rewards + (1 - dones) * self.gamma * self.target_agent.get_max_q(next_states)
            else:
                target = rewards + (1 - dones) * self.gamma * self.agent.get_max_q(next_states)
            q = self.agent.get_q(states, actions)
            
            td_error = target - q

            loss = torch.mean(td_error**2)
            
            # do gradient update to the agent 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        if self.use_target:
            if self.target_update_method == "soft":
                self._target_soft_update()
            elif self.interaction_step%self.target_update_interval==0:
                self._target_hard_update()


        result.add_metric("network/loss", loss.item())
        result.add_metric("td_error", td_error.mean().item())
        result.add_metric("q_value_mean", q.mean().item())
       

        self.gradient_step += 1
        return result

    def random_choose_action(self):
        """You can use this function to implement epsilon-greedy exploration strategy"""
        if self.epsilon_schedular=="linear":
            epsilon = max(self.end_epsilon, self.start_epsilon - self.interaction_step / (self.total_epoch * self.interact_per_epoch) * (self.start_epsilon - self.end_epsilon))
        else:
            epsilon = self.start_epsilon
        
        return np.random.rand() < epsilon