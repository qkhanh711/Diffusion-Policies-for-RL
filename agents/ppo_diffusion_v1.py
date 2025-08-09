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
                 clip_param=0.2,
                 beta_schedule='linear',
                 n_timesteps=10,  # More timesteps for better action quality
                 ema_decay=0.99,  # Faster EMA for more exploration
                 step_start_ema=500,  # Start EMA earlier
                 update_ema_every=3,  # More frequent EMA updates
                 lr=2e-4,  # Higher learning rate for faster learning
                 lr_decay=True,
                 lr_maxt=10000,
                 grad_norm=1.0,
                 entropy_coef=0.02,  # Higher entropy for more exploration
                 value_loss_coef=0.25,  # Lower value loss weight
                 warmup_steps=200,  # Shorter warmup
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.warmup_steps = warmup_steps

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr*3, betas=(0.9, 0.999))  # Critic learns much faster

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr*0.1)  # Don't decay too much
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr*0.3)

        self.entropy_coef = entropy_coef
        self.value_coef = value_loss_coef
        self.initial_lr = lr
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param  # Use standard PPO clip
        self.device = device

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)


    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'actor_loss': [], 'entropy': [], 'kl_div': [], 'clipfrac': []}
        
        for iteration in range(iterations):
            try:
                # Sample batch
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                """ PPO Loss Implementation (from diffusion_ppo.py) """
                # Compute returns and current values
                with torch.no_grad():
                    next_values = self.critic(next_state).squeeze()
                    returns = reward.squeeze() + self.gamma * next_values * not_done.squeeze()
                    old_values = self.critic(state).squeeze()
                
                # Compute advantages
                advantages = returns - old_values
                
                # Normalize advantages (as in diffusion_ppo)
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Clip advantages by 5th and 95th percentile (from diffusion_ppo)
                advantage_min = torch.quantile(advantages, 0.05)
                advantage_max = torch.quantile(advantages, 0.95)
                advantages = advantages.clamp(min=advantage_min, max=advantage_max)
                
                # Get old log probabilities (approximate using current model)
                with torch.no_grad():
                    old_logprobs = -self.actor.loss(action, state).detach()  # Convert loss to logprob
                    old_logprobs = old_logprobs.clamp(min=-5, max=2)
                
                # Get new log probabilities
                new_logprobs = -self.actor.loss(action, state)  # Convert loss to logprob
                new_logprobs = new_logprobs.clamp(min=-5, max=2)
                
                # Compute ratio
                logratio = new_logprobs - old_logprobs
                ratio = logratio.exp()
                
                # PPO clipped policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Entropy loss (encourage exploration)
                entropy_loss = -self.entropy_coef * new_logprobs.mean()
                
                # Value loss with optional clipping
                new_values = self.critic(state).squeeze()
                v_loss_unclipped = (new_values - returns) ** 2
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -0.2, 0.2  # Standard PPO value clipping
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean()
                
                # Total actor loss
                total_actor_loss = policy_loss + entropy_loss
                
                # Compute diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                
                # Update actor
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.actor_optimizer.step()
                
                # Update critic (separate optimization)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                self.critic_optimizer.step()

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                self.step += 1

                # Log metrics
                metric['ppo_loss'].append(float(policy_loss.item()) if not torch.isnan(policy_loss) else 0.0)
                metric['value_loss'].append(float(value_loss.item()) if not torch.isnan(value_loss) else 0.0)
                metric['actor_loss'].append(float(total_actor_loss.item()) if not torch.isnan(total_actor_loss) else 0.0)
                metric['entropy'].append(float(-entropy_loss.item()) if not torch.isnan(entropy_loss) else 0.0)
                metric['kl_div'].append(float(approx_kl.item()) if not torch.isnan(approx_kl) else 0.0)
                metric['clipfrac'].append(clipfrac)

            except Exception as e:
                print(f"Error in PPO training iteration {iteration}: {e}")
                metric['ppo_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)
                metric['entropy'].append(0.0)
                metric['kl_div'].append(0.0)
                metric['clipfrac'].append(0.0)
                continue

        # Learning rate scheduling (more conservative)
        if self.lr_decay and self.step % 200 == 0:  # Even less frequent LR updates
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            # Mix exploration: Sometimes use main model for exploration
            if self.step > self.step_start_ema:
                # 80% EMA (stable), 20% main model (exploration)
                if np.random.random() < 0.8:
                    action = self.ema_model.sample(state)
                else:
                    action = self.actor.sample(state)
            else:
                action = self.actor.sample(state)
                
            # Add small noise for exploration in early training
            if self.step < 2000:
                noise_scale = 0.1 * (1.0 - self.step / 2000.0)  # Decreasing noise
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.clamp(action, -self.max_action, self.max_action)
                
        return action.cpu().data.numpy().flatten()

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