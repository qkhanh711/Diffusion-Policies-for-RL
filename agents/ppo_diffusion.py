# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

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
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)  # Output single value function
        )

    def forward(self, state):
        return self.model(state)

    def q1(self, state):
        return self.model(state)

    def q_min(self, state):
        return self.model(state)


class Diffusion_PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 gamma=0.99,
                 tau=0.95,
                 clip_param=0.005,
                 beta_schedule='linear',
                 n_timesteps=2,
                 ema_decay=0.995,
                 step_start_ema=500,
                 update_ema_every=3,
                 lr=1e-4,
                 lr_decay=True,
                 lr_maxt=2000,
                 grad_norm=0.5,  # Much smaller grad norm
                 entropy_coef=0.01,  # Add entropy regularization
                 value_loss_coef=0.5,  # Smaller value loss weight
                 warmup_steps=1000,  # More warmup steps
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.warmup_steps = warmup_steps

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.entropy_coef = entropy_coef
        # self.value_loss_coef = value_loss_coef
        self.initial_lr = lr
        self.value_coef = value_loss_coef
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_param = min(clip_param, 0.1)  # Smaller clip to prevent explosion
        self.device = device

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def compute_gae(self, rewards, values, dones, next_value):
        values = np.append(values, next_value)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def estimate_log_prob(self, state, action, num_samples=20, bandwidth=0.1):
        """
        Estimate log probability log π(a|s) from diffusion model by kernel density estimation
        """
        with torch.no_grad():
            state_rpt = torch.repeat_interleave(state, repeats=num_samples, dim=0)
            samples = self.actor.sample(state_rpt).reshape(num_samples, -1)

            diffs = samples - action  # [num_samples, action_dim]
            squared = torch.sum(diffs ** 2, dim=1)  # [num_samples]
            log_probs = -squared / (2 * bandwidth ** 2)
            log_prob = torch.logsumexp(log_probs, dim=0) - np.log(num_samples)
            return -log_prob.mean()

    def batch_log_prob(self, state_batch, action_batch, num_samples=20, bandwidth=0.1):
        B = state_batch.size(0)
        state_rpt = state_batch.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, self.state_dim)  # [B * num_samples, state_dim]
        action_batch = action_batch  # [B, action_dim]

        with torch.no_grad():
            samples = self.actor.sample(state_rpt).reshape(B, num_samples, -1)
            diffs = samples - action_batch.unsqueeze(1)  # [B, num_samples, action_dim]
            sq = torch.sum(diffs ** 2, dim=2)  # [B, num_samples]
            log_probs = -sq / (2 * bandwidth ** 2)
            log_prob = torch.logsumexp(log_probs, dim=1) - np.log(num_samples)
            return log_prob  # [B]


    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'actor_loss': []}
        
        # Learning rate warm-up
        if self.step < self.warmup_steps:
            lr_scale = self.step / self.warmup_steps
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.initial_lr * lr_scale
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.initial_lr * 2 * lr_scale
        
        for iteration in range(iterations):
            try:
                # Sample batch
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                with torch.no_grad():
                    values = self.critic(state).squeeze()
                    next_values = self.critic(next_state).squeeze()
                    
                    # Compute TD targets and advantages
                    returns = reward + self.gamma * next_values * not_done
                    advantages = returns - values
                    
                    # Get old log probabilities with safety
                    actor_loss_raw = self.actor.loss(action, state)
                    # log_probs_old = -torch.clamp(actor_loss_raw, min=-10.0, max=10.0).detach()
                    # log_probs_old = torch.stack([
                        # self.estimate_log_prob(state[i].unsqueeze(0), action[i].unsqueeze(0))
                        # for i in range(batch_size)
                    # ])
                    log_probs_old = self.batch_log_prob(state, action)



                # Normalize advantages more carefully
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                advantages = torch.clamp(advantages, -5.0, 5.0)

                """ Policy Training with multiple epochs """
                total_policy_loss = 0
                num_policy_updates = 3  # Multiple updates per iteration
                
                for epoch in range(num_policy_updates):
                    actor_loss_new = self.actor.loss(action, state)
                    # new_log_probs = -torch.clamp(actor_loss_new, min=-10.0, max=10.0)
                    # new_log_probs = torch.stack([
                        # self.estimate_log_prob(state[i].unsqueeze(0), action[i].unsqueeze(0))
                        # for i in range(batch_size)
                    # ])
                    new_log_probs = self.batch_log_prob(state, action)


                    # Compute ratio with careful bounds
                    log_ratio = new_log_probs - log_probs_old
                    log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
                    ratio = torch.exp(log_ratio)
                    
                    # PPO clipped loss
                    ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
                    surrogate1 = ratio * advantages.detach()
                    surrogate2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages.detach()
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    # Add entropy bonus if possible
                    try:
                        # Approximate entropy using action variance
                        action_std = torch.std(action, dim=0).mean()
                        entropy_bonus = -self.entropy_coef * torch.log(action_std + 1e-8)
                        policy_loss += entropy_bonus
                    except:
                        pass

                    # Check for valid loss
                    if torch.isnan(policy_loss) or torch.isinf(policy_loss) or abs(policy_loss.item()) > 1000:
                        # print(f"Warning: Invalid policy_loss {policy_loss} at epoch {epoch}, skipping")
                        continue

                    # Early stopping if KL divergence too high
                    with torch.no_grad():
                        kl_div = 0.5 * ((log_ratio) ** 2).mean()
                        if kl_div > 0.02:  # Stop if KL too high
                            # print(f"Early stopping due to high KL: {kl_div}")
                            break

                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    
                    # Adaptive gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
                    
                    self.actor_optimizer.step()
                    total_policy_loss += policy_loss.item()

                """ Value Function Training """
                value_pred = self.critic(state).squeeze()
                if value_pred.dim() != returns.dim():
                    returns = returns.squeeze()
                
                value_loss = F.mse_loss(value_pred, returns.detach()) * self.value_coef

                if torch.isnan(value_loss) or torch.isinf(value_loss) or abs(value_loss.item()) > 1e6:
                    # print(f"Warning: Invalid value_loss {value_loss}, skipping")
                    metric['ppo_loss'].append(total_policy_loss / num_policy_updates)
                    metric['value_loss'].append(0.0)
                    metric['actor_loss'].append(total_policy_loss / num_policy_updates)
                    continue

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm * 2)
                self.critic_optimizer.step()

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                self.step += 1

                # Log metrics
                metric['ppo_loss'].append(total_policy_loss / num_policy_updates)
                metric['value_loss'].append(float(value_loss.item()))
                metric['actor_loss'].append(total_policy_loss / num_policy_updates)

            except Exception as e:
                print(f"Error in training iteration {iteration}: {e}")
                metric['ppo_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)
                continue

        # Learning rate scheduling
        if self.lr_decay and self.step > self.warmup_steps:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        # Adaptive number of samples based on training progress
        num_samples = max(10, 50 - self.step // 100)  # Giảm dần từ 50 xuống 10
        state_rpt = torch.repeat_interleave(state, repeats=num_samples, dim=0)
        
        with torch.no_grad():
            # Use EMA model when available for more stable actions
            if self.step > self.step_start_ema:
                action = self.ema_model.sample(state_rpt)
            else:
                action = self.actor.sample(state_rpt)
            
            # Improved value-guided selection with temperature
            values = self.critic(state_rpt).flatten()
            
            # Add noise for exploration early in training
            if self.step < 2000:
                noise = torch.randn_like(values) * 0.1
                values = values + noise
            
            # Temperature-based softmax
            temperature = max(0.1, 1.0 - self.step / 5000.0)
            probs = F.softmax(values / temperature, dim=0)
            idx = torch.multinomial(probs, 1)
        
        return action[idx].cpu().data.numpy().flatten()

    # def sample_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     with torch.no_grad():
    #         # Giảm số lượng samples để tăng tốc
    #         action = self.actor.sample(state)
    #     return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
