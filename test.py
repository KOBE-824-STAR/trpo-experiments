import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5')
print(env.observation_space)