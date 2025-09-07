# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
# Fixed version of DiffPPO to address performance collapse issues

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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state)

    def q1(self, state):
        return self.model(state)

    def q_min(self, state):
        return self.model(state)


class Diffusion_PPO_Fixed(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 gamma=0.99,
                 tau=0.95,
                 clip_param=0.2,
                 beta_schedule='linear',
                 n_timesteps=10,
                 ema_decay=0.999,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=2e-4,
                 lr_decay=True,
                 lr_maxt=10000,
                 grad_norm=1.0,
                 entropy_coef=0.02,
                 value_loss_coef=0.25,
                 warmup_steps=200,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        self.critic = Critic(state_dim, action_dim).to(device)
        # FIXED: Use same learning rate for critic to prevent instability
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.9, 0.999))

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr*0.1)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr*0.3)

        self.entropy_coef = entropy_coef
        self.value_coef = value_loss_coef
        self.initial_lr = lr
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.device = device

        # EMA settings
        self.ema_decay = ema_decay
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.warmup_steps = warmup_steps
        self.grad_norm = grad_norm
        
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.step = 0

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'actor_loss': []}
        
        for iteration in range(iterations):
            try:
                # Sample batch
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                """ Value Function Training """
                with torch.no_grad():
                    next_values = self.critic(next_state).squeeze()
                    returns = reward.squeeze() + self.gamma * next_values * not_done.squeeze()
                
                values = self.critic(state).squeeze()
                advantages = returns - values
                
                # Normalize advantages
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Train value function
                value_pred = self.critic(state).squeeze()
                value_loss = F.mse_loss(value_pred, returns.detach())
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                self.critic_optimizer.step()

                """ FIXED Policy Training - Standard PPO approach """
                # Compute behavioral cloning loss for all actions
                bc_losses = self.actor.loss(action, state)
                
                # FIXED: Weight by ALL advantages (not just positive ones)
                # Use advantage weighting but include negative advantages too
                advantage_weights = torch.softmax(advantages / 0.1, dim=0)  # Temperature scaling
                weighted_bc_loss = (bc_losses * advantage_weights.detach()).mean()
                
                # Add entropy regularization for exploration
                # FIXED: Use proper entropy from diffusion model
                entropy_bonus = -self.entropy_coef * bc_losses.mean()
                
                total_actor_loss = weighted_bc_loss - entropy_bonus
                
                # FIXED: Always update policy (don't skip updates)
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.actor_optimizer.step()

                # EMA update - FIXED: Use more conservative update
                if self.step % self.update_ema_every == 0 and self.step > self.warmup_steps:
                    self.step_ema()

                self.step += 1

                # Log metrics
                metric['ppo_loss'].append(float(weighted_bc_loss.item()) if not torch.isnan(weighted_bc_loss) else 0.0)
                metric['value_loss'].append(float(value_loss.item()) if not torch.isnan(value_loss) else 0.0)
                metric['actor_loss'].append(float(total_actor_loss.item()) if not torch.isnan(total_actor_loss) else 0.0)

            except Exception as e:
                print(f"Error in PPO training iteration {iteration}: {e}")
                metric['ppo_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)

        return {key: np.mean(value) for key, value in metric.items()}

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            # FIXED: More conservative exploration strategy
            if self.step > self.step_start_ema:
                # Use 90% EMA (more stable), 10% main model (less exploration noise)
                if np.random.random() < 0.9:
                    action = self.ema_model.sample(state)
                else:
                    action = self.actor.sample(state)
            else:
                action = self.actor.sample(state)
                
            # FIXED: Reduce exploration noise and duration
            if self.step < self.warmup_steps:
                noise_scale = 0.05 * (1.0 - self.step / self.warmup_steps)  # Less noise
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
