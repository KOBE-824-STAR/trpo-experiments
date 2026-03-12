from utils.result import Result
import torch
import numpy as np 
import gymnasium as gym 


from agent.atrai_agent import AgentBase
from .utils import atari_to_useful_action

def evaluate_atari(agent:AgentBase, envs: gym.Env, test_episodes:int):
    """
    Evaluate the policy in the given environments.
    
    Args:
        agent: The RL agent to be evaluated.
        envs: Vec environments to evaluate on.
        test_episodes: Number of episodes to run for each environment.
    """
    agent.eval()
    states = envs.reset()
    episode_rewards = []
    envs_rewards = np.zeros((envs.num_envs,),dtype=np.float32)
    with Result(head="test") as result:
        while len(episode_rewards)<test_episodes:
            with torch.no_grad():
                # actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
                actions = agent.select_action(states,deterministic=True) # test with deterministic policy
            next_states, rewards, terminateds, infos = envs.step(atari_to_useful_action(actions))
            envs_rewards += rewards
            for i in range(envs.num_envs):
                if infos[i]['lives']==0: # TODO: In game like MontezumaRevenge, it is always lives=0 and we need extra process
                    episode_rewards.append(envs_rewards[i])
                    envs_rewards[i] = 0.0
            states = next_states

    result.add_metric("episode_rewards_mean", np.mean(episode_rewards))
    result.add_metric("episode_rewards_std", np.std(episode_rewards))

    return result
        
    