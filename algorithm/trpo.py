import gymnasium as gym
import torch 
from torch.distributions import kl_divergence
import numpy as np
from typing import Tuple

from logger.logger import Logger
from memory.memory import ReplayBuffer, TrajectoryRollout
from agent.agent import AgentBase
from utils.result import Result
from utils.utils import get_flat_grad, conjugate_gradients, kl_product, get_model_flat_parameters, load_flat_parameters_to_model, OPTIMIZER_DICT
from .baseonpolicy import OnPolicyAlgorithm

class TRPO(OnPolicyAlgorithm):

    """Base class for off-policy RL algorithms."""


    def __init__(self, training_envs:gym.Env, testing_envs:gym.Env, buffer: ReplayBuffer | TrajectoryRollout, agent: AgentBase, logger: Logger, device, save_pth: str, best_pth:str, args):
        super(TRPO,self).__init__(training_envs, testing_envs, buffer, agent, logger, device, save_pth, best_pth, args)

        algo_args = args.algorithm

        self.delta = algo_args.delta
        self.cg_steps = algo_args.cg_steps
        self.linesearch_coeffient = algo_args.linesearch_coeffient
        self.backtracking_steps = algo_args.backtracking_steps
        self.lr = algo_args.learning_rate
        self.lambda_ = algo_args.lambda_
        self.optimizer: torch.optim.Optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.critic.parameters(), lr=self.lr)
        self.advan_norm = algo_args.advan_norm
        self.critic_update_steps = algo_args.critic_update_steps
        self.critic_batch_size = algo_args.critic_batch_size
        

    def _update_buffer(self, batch):
        self.buffer.add(batch)
    
    def compute_advantages_from_traj(self)->Tuple[np.ndarray,np.ndarray]:
        states = self.traj_rollout.states
        rewards = self.traj_rollout.rewards
        dones = self.traj_rollout.dones
        masks = self.traj_rollout.masks

        trajnum, max_episode_length = rewards.shape
        flat_states = states.reshape(trajnum * max_episode_length, *self.observation_space.shape)

        with torch.no_grad():
            values = self.agent.get_value(flat_states)
            values = values.reshape(trajnum, max_episode_length).cpu().numpy()

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = np.zeros(trajnum, dtype=np.float32)

        for step in reversed(range(max_episode_length)):
            if step == max_episode_length - 1:
                next_values = np.zeros(trajnum, dtype=np.float32)
            else:
                next_values = values[:, step + 1]

            non_terminal = 1.0 - dones[:, step]
            delta = rewards[:, step] + self.gamma * next_values * non_terminal - values[:, step]
            last_gae_lam = delta + self.gamma * self.lambda_ * non_terminal * last_gae_lam
            advantages[:, step] = last_gae_lam * masks[:, step]

        returns = advantages + values
        valid_steps = masks.astype(bool)
        return advantages[valid_steps], returns[valid_steps]
        

    def compute_advantages_from_rollout(self)->Tuple[np.ndarray,np.ndarray]:
        # TODO: correctly compute the advantages and returns from the rollout buffer
        states = self.buffer.buffer['states']
        rewards = self.buffer.buffer['rewards']
        dones = self.buffer.buffer['dones']

        rollout_length = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
        states = states[:, :rollout_length]
        rewards = rewards[:, :rollout_length]
        dones = dones[:, :rollout_length]

        num_envs = rewards.shape[0]
        flat_states = states.reshape(num_envs * rollout_length, *self.observation_space.shape)

        with torch.no_grad():
            values = self.agent.get_value(flat_states)
            values = values.reshape(num_envs, rollout_length).cpu().numpy()
            last_values = self.agent.get_value(self.observations).squeeze(1).cpu().numpy()

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = np.zeros(num_envs, dtype=np.float32)

        for step in reversed(range(rollout_length)):
            if step == rollout_length - 1:
                next_values = last_values
            else:
                next_values = values[:, step + 1]

            non_terminal = 1.0 - dones[:, step]
            delta = rewards[:, step] + self.gamma * next_values * non_terminal - values[:, step]
            last_gae_lam = delta + self.gamma * self.lambda_ * non_terminal * last_gae_lam
            advantages[:, step] = last_gae_lam

        returns = advantages + values
        return advantages, returns
        

        
    def _update_policy(self):
        # TODO: implement the updating procedure according to the TRPO algorithm 
        with Result("train") as result:

        # 1.compute the advantages and returns from the trajectory or rollout buffer
            if self.collect_traj:
                advantages, returns = self.compute_advantages_from_traj()
                valid_steps = self.traj_rollout.masks.astype(bool)
                states = self.traj_rollout.states[valid_steps]
                actions = self.traj_rollout.actions[valid_steps]
                old_log_probs = self.traj_rollout.log_probs[valid_steps]
            else:
                advantages, returns = self.compute_advantages_from_rollout()
                rollout_length = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
                states = self.buffer.buffer['states'][:, :rollout_length].reshape(-1, *self.observation_space.shape)
                actions = self.buffer.buffer['actions'][:, :rollout_length].reshape(-1, self.action_dim)
                old_log_probs = self.buffer.buffer['log_probs'][:, :rollout_length].reshape(-1)

            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=self.device)
            raw_advantages = advantages.clone()

        # remember to do the advantage normalization to stabilize the training 
            if self.advan_norm:
                adv_std = advantages.std(unbiased=False)
                if torch.isfinite(adv_std).item() and adv_std.item() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()


        # 2.use conjugate_gradients to compute the gradient direction and stepsize (some mathematical methods are implemented in utils/utils.py)
            with torch.no_grad():
                old_mu, old_std = self.agent.actor(states)
                old_mu, old_std = old_mu.detach(), old_std.detach()

            def surrogate_target():
                log_probs = self.agent.log_prob(states, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                return torch.mean(ratio * advantages)

            def mean_kl():
                mu, std = self.agent.actor(states)
                kl = torch.log(std / old_std) + (old_std.pow(2) + (old_mu - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
                return torch.sum(kl, dim=1).mean()

            old_surrogate_target = surrogate_target().detach()
            with torch.no_grad():
                initial_log_probs = self.agent.log_prob(states, actions)
                initial_ratio = torch.exp(initial_log_probs - old_log_probs)
                initial_log_prob_diff = initial_log_probs - old_log_probs

            policy_gradient = get_flat_grad(surrogate_target(), self.agent.actor, retain_graph=True).detach()
            policy_gradient_norm = torch.norm(policy_gradient, p=2)
            actor_update_valid = torch.isfinite(policy_gradient_norm).item() and policy_gradient_norm.item() > 1e-12

            kl_grad = get_flat_grad(mean_kl(), self.agent.actor, create_graph=True)
            if actor_update_valid:
                gradient_direction = conjugate_gradients(self.agent.actor, policy_gradient, kl_grad, self.cg_steps)
                xHx = torch.sum(gradient_direction * kl_product(gradient_direction, kl_grad, self.agent.actor))
                if not torch.isfinite(xHx) or xHx <= 1e-12:
                    actor_update_valid = False
            if actor_update_valid:
                stepsize = torch.sqrt(2.0 * self.delta / xHx)
            else:
                gradient_direction = torch.zeros_like(policy_gradient)
                xHx = torch.zeros((), device=self.device)
                stepsize = torch.zeros((), device=self.device)
        

        # 3.linesearch to find the best stepsize along the gradient direction that can satisfy the KL constraint and improve the surrogate target
            old_parameter = get_model_flat_parameters(self.agent.actor).detach()
            new_parameter = old_parameter.clone()
            new_surrogate_target = old_surrogate_target
            new_kl = torch.zeros((), device=self.device)
            backtrack_step = self.backtracking_steps
            accepted = False

            if actor_update_valid:
                for step in range(self.backtracking_steps):
                    backtrack_coeffient = self.linesearch_coeffient ** step
                    candidate_parameter = old_parameter + backtrack_coeffient * stepsize * gradient_direction
                    load_flat_parameters_to_model(candidate_parameter, self.agent.actor)

                    with torch.no_grad():
                        candidate_surrogate_target = surrogate_target()
                        candidate_kl = mean_kl()

                    if (
                        torch.isfinite(candidate_kl)
                        and torch.isfinite(candidate_surrogate_target)
                        and candidate_kl <= self.delta
                        and candidate_surrogate_target > old_surrogate_target
                    ):
                        new_parameter = candidate_parameter
                        new_surrogate_target = candidate_surrogate_target
                        new_kl = candidate_kl
                        backtrack_step = step
                        accepted = True
                        break

            load_flat_parameters_to_model(new_parameter if accepted else old_parameter, self.agent.actor)

        # 4.update the value function with MSE loss, remember to sample batch and update self.critic_update_steps epochs
            value_loss = torch.zeros((), device=self.device)
            data_size = states.shape[0]
            for _ in range(self.critic_update_steps):
                batch_indices = torch.randint(0, data_size, (self.critic_batch_size,), device=self.device)
                value = self.agent.get_value(states[batch_indices]).squeeze(1)
                value_loss = torch.mean((value - returns[batch_indices]) ** 2)
                self.optimizer.zero_grad()
                value_loss.backward()
                self.optimizer.step()

            self.gradient_step += 1



        result.add_metric("value/loss", value_loss.item())
        result.add_metric("actor/backtrack_step",backtrack_step)
        result.add_metric("actor/accepted",float(accepted))
        result.add_metric("actor/old_surrogate_target",old_surrogate_target.item())
        result.add_metric("actor/new_surrogate_target",new_surrogate_target.item())
        result.add_metric("actor/new_kl",new_kl.item())
        result.add_metric("actor/raw_adv_mean", raw_advantages.mean().item())
        result.add_metric("actor/raw_adv_std", raw_advantages.std(unbiased=False).item())
        result.add_metric("actor/raw_adv_min", raw_advantages.min().item())
        result.add_metric("actor/raw_adv_max", raw_advantages.max().item())
        result.add_metric("actor/adv_mean", advantages.mean().item())
        result.add_metric("actor/adv_std", advantages.std(unbiased=False).item())
        result.add_metric("actor/adv_min", advantages.min().item())
        result.add_metric("actor/adv_max", advantages.max().item())
        result.add_metric("actor/old_log_prob_mean", old_log_probs.mean().item())
        result.add_metric("actor/old_log_prob_std", old_log_probs.std().item())
        result.add_metric("actor/log_prob_mean", initial_log_probs.mean().item())
        result.add_metric("actor/log_prob_std", initial_log_probs.std().item())
        result.add_metric("actor/log_prob_diff_abs_max", initial_log_prob_diff.abs().max().item())
        result.add_metric("actor/ratio_mean", initial_ratio.mean().item())
        result.add_metric("actor/ratio_std", initial_ratio.std().item())
        result.add_metric("actor/policy_gradient_l2norm",policy_gradient_norm.item())
        result.add_metric("actor/xHx",xHx.item())
        result.add_metric("actor/stepsize",stepsize.item())
        result.add_metric("actor/gradient_direction_l2norm",torch.norm(gradient_direction,p=2).item())
        result.add_metric("actor/new_param_l2norm",torch.norm(new_parameter).item())
        return result


                
