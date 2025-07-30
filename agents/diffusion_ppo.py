import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),                           
                                      nn.Linear(hidden_dim, 1))
        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),                          
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
        
    def forward(self, state, action):
        return self.q1_model(torch.cat([state, action], dim=-1)), self.q2_model(torch.cat([state, action], dim=-1))
    
    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)
    
    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 gamma=0.99,
                 tau=0.95,
                 clip_ratio=0.2,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 n_timesteps=100,
                 lr=3e-4,
                 grad_norm=1.0):

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.grad_norm = grad_norm

        # Actor + Critic
        self.model = MLP(state_dim, action_dim, device)
        self.actor = Diffusion(state_dim, action_dim, self.model, max_action, n_timesteps=n_timesteps).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, device=self.device)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'actor_loss': [], 'critic_loss': [], 'ppo_loss': []}  # Thêm ppo_loss vào

        for _ in range(iterations):
            # Sample batch from replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Estimate values """
            with torch.no_grad():
                value = self.critic.q1(state, action).squeeze()
                next_value = self.critic.q1(next_state, action).squeeze()
                advantage = reward + self.gamma * not_done * next_value - value
                returns = advantage + value

            """ Critic Update """
            current_value = self.critic.q1(state, action).squeeze()
            critic_loss = F.mse_loss(current_value, returns)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optim.step()

            """ Policy Update (PPO-style) """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            # Estimate new log prob
            log_prob = self.actor.get_log_prob(state, new_action)
            old_log_prob = self.actor.get_log_prob(state, action).detach()

            ratio = torch.exp(log_prob - old_log_prob)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
            policy_loss = -torch.min(ratio * advantage, clip_adv).mean()

            entropy = -log_prob.mean()
            actor_loss = bc_loss + policy_loss - self.ent_coef * entropy

            # PPO Loss
            ppo_loss = policy_loss + self.ent_coef * entropy

            self.actor_optim.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optim.step()

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('PPO Loss', ppo_loss.item(), self.step)  # Thêm PPO Loss vào
                log_writer.add_scalar('PPO Policy Loss', policy_loss.item(), self.step)
                log_writer.add_scalar('Entropy', entropy.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Advantage Mean', advantage.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['ppo_loss'].append(ppo_loss.item())  # Lưu PPO Loss vào metric
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric


    def sample_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.actor.sample(state)

    def save_model(self, dir, id=None):
        self.actor.save(dir, id)
        self.critic.save(dir, id)
