from utils.result import Result
import torch
import numpy as np 

def evaluate_atari(agent, envs, test_episodes):
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
    with Result() as result:
        while len(episode_rewards)<test_episodes:
            with torch.no_grad():
                actions = agent.get_action(states,deterministic=True) # test with deterministic policy
            next_states, rewards, terminateds, infos = envs.step(actions)
            envs_rewards += rewards
            for i in range(envs.num_envs):
                if terminateds[i]: # this is because the testing_envs is not wrapped with EpisodeLife, so we need to check the terminated flag to get the episode rewards
                    episode_rewards.append(envs_rewards[i])
                    envs_rewards[i] = 0.0
            states = next_states

    result.add_metric("episode_rewards_mean", np.mean(episode_rewards))
    result.add_metric("episode_rewards_std", np.std(episode_rewards))

    return result
        
    