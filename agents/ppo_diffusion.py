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
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.value_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.value_model(state)


class Diffusion_PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount=0.99,
                 tau=0.005,
                 beta_schedule='linear',
                 n_timesteps=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 clip_ratio=0.2,
                 value_clip_ratio=0.2,
                 norm_adv=True,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        
        # Initialize basic Diffusion model for actor
        self.actor = Diffusion(
            state_dim=state_dim, 
            action_dim=action_dim, 
            model=self.model, 
            max_action=max_action,
            beta_schedule=beta_schedule, 
            n_timesteps=n_timesteps
        ).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Value function
        self.critic = Critic(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.norm_adv = norm_adv

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'entropy_loss': [], 'total_loss': []}
        
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            
            # Compute returns and advantages
            with torch.no_grad():
                values = self.critic(state).squeeze()
                next_values = self.critic(next_state).squeeze()
                returns = reward + self.discount * not_done.squeeze() * next_values
                advantages = returns - values
                
                # Normalize advantages
                if self.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO loss computation - simplified version
            # Compute policy loss using actor's loss method (behavior cloning)
            policy_loss = self.actor.loss(action, state)
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns.detach())
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            metric['ppo_loss'].append(policy_loss.item())
            metric['value_loss'].append(value_loss.item())
            metric['total_loss'].append(total_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
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
