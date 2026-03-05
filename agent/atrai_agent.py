from utils.networks import AtariDQNNetwork
import torch.nn as nn
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space 
from typing import Union


def to_correct_device_tensor(input, device)-> torch.Tensor:
    if isinstance(input, np.ndarray):
        return torch.tensor(input,dtype=torch.float32,device=device)
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        return input
    else:
        raise TypeError("input must be np array or torch tensor")


class AgentBase(nn.Module, ABC):
    def __init__(self,observation_space: Space, action_space: Space, device):
        super(AgentBase, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space 
        self.device = device
        
    
    @abstractmethod
    def select_action(self, states:Union[np.ndarray, torch.Tensor], deterministic:bool) -> np.ndarray:
        """Choose the action according to the states agent observed. """
        pass 
    
    def _get_other_parameters(self):
        return dict(
            observation_space = self.observation_space,
            action_space = self.action_space,
            device = self.device 
        )

    def save(self, pth_file: str):
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_other_parameters()}, pth_file)
        
    def load(self,path:str):
        pth_file = torch.load(path,weights_only=False)
        self.load_state_dict(pth_file['state_dict'])
        # TODO: check when other parameters are added to Agent
        data = pth_file['data']
        self.observation_space = data['observation_space']
        self.action_space = data['action_space']
        self.device = data['device']
        


class AtariDQNAgent(AgentBase):
    def __init__(self, observation_space: Space, action_space: Space, device):
        super(AtariDQNAgent, self).__init__(observation_space, action_space, device)
        self.num_actions = action_space.n 
        self.observation_shape = observation_space.shape
        self.network = AtariDQNNetwork(observation_space.shape, action_space.n)

    

    def select_action(self, state:Union[np.ndarray, torch.Tensor], deterministic=False) -> np.ndarray:
        """
        params: state: (batch, channel, 84,84)
        output: action: (batch, 1)
        """
        state = to_correct_device_tensor(state, self.device)

        all_q_values = self.network(state)
        
        if deterministic:
            action = torch.argmax(all_q_values, dim=1,keepdim=True)
        else:
            action = torch.multinomial(torch.softmax(all_q_values, dim=1), num_samples=1)

        return action.cpu().numpy()

    def get_q(self, states:Union[np.ndarray, torch.Tensor], actions:Union[np.ndarray, torch.Tensor]):
        states, actions = to_correct_device_tensor(states, self.device) ,to_correct_device_tensor(actions, self.device)
        all_q_values = self.network(states)
        q_values = all_q_values.gather(1, actions)
        return q_values 

    def get_max_q(self, states:Union[np.ndarray, torch.Tensor]):

        states = to_correct_device_tensor(states, self.device)
        all_q_values = self.network(states)
        q_max = torch.max(all_q_values, dim=1, keepdim=True).values
        return q_max 
        


