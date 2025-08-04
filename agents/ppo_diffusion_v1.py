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
                 n_timesteps=5,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=10,  # Less frequent EMA updates
                 lr=5e-5,  # Very conservative learning rate
                 lr_decay=True,
                 lr_maxt=10000,
                 grad_norm=0.5,  # Smaller grad norm for stability
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 warmup_steps=500,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)  # Add weight decay

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.warmup_steps = warmup_steps

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr*2, weight_decay=1e-5)  # Critic learns faster

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr*0.1)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr*0.2)

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
        
        # Track best experiences for supervised learning
        self.best_experiences = []
        self.max_best_size = 1000
        self.update_frequency = 0

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
    
    def add_best_experience(self, state, action, advantage):
        """Add good experiences to supervised learning buffer"""
        if advantage > 0:  # Only store positive advantage experiences
            experience = {
                'state': state.clone().detach(),
                'action': action.clone().detach(),
                'advantage': advantage.item()
            }
            self.best_experiences.append(experience)
            
            # Keep only best experiences
            if len(self.best_experiences) > self.max_best_size:
                # Sort by advantage and keep top experiences
                self.best_experiences.sort(key=lambda x: x['advantage'], reverse=True)
                self.best_experiences = self.best_experiences[:self.max_best_size]


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
                for _ in range(2):  # Reduce iterations
                    value_pred = self.critic(state).squeeze()
                    value_loss = F.mse_loss(value_pred, returns.detach())
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                    self.critic_optimizer.step()

                """ Collect Best Experiences """
                for i in range(len(advantages)):
                    if advantages[i] > 0.5:  # Only very good experiences
                        self.add_best_experience(state[i:i+1], action[i:i+1], advantages[i])

                """ Policy Training - Supervised Learning from Best Experiences """
                policy_loss = torch.tensor(0.0)
                
                # Train on current good experiences
                good_mask = advantages > 0
                if good_mask.sum() > 0:
                    good_states = state[good_mask]
                    good_actions = action[good_mask]
                    
                    # Simple behavioral cloning on good actions
                    bc_loss = self.actor.loss(good_actions, good_states).mean()
                    policy_loss = bc_loss
                
                # Train on historical best experiences (every few steps)
                if len(self.best_experiences) > 50 and self.update_frequency % 5 == 0:
                    # Sample from best experiences
                    num_samples = min(32, len(self.best_experiences))
                    sampled_experiences = np.random.choice(self.best_experiences, num_samples, replace=False)
                    
                    best_states = torch.cat([exp['state'] for exp in sampled_experiences], dim=0)
                    best_actions = torch.cat([exp['action'] for exp in sampled_experiences], dim=0)
                    
                    # Supervised learning on best experiences
                    supervised_loss = self.actor.loss(best_actions, best_states).mean()
                    policy_loss = policy_loss + 0.5 * supervised_loss

                # Update policy if we have a valid loss
                if policy_loss > 0 and not torch.isnan(policy_loss):
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                    self.actor_optimizer.step()

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                self.step += 1
                self.update_frequency += 1

                # Log metrics
                metric['ppo_loss'].append(float(policy_loss.item()) if not torch.isnan(policy_loss) else 0.0)
                metric['value_loss'].append(float(value_loss.item()) if not torch.isnan(value_loss) else 0.0)
                metric['actor_loss'].append(float(policy_loss.item()) if not torch.isnan(policy_loss) else 0.0)

            except Exception as e:
                print(f"Error in PPO training iteration {iteration}: {e}")
                metric['ppo_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)
                continue

        # Learning rate scheduling (very conservative)
        if self.lr_decay and self.step % 500 == 0:  # Very infrequent updates
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            # Use EMA model when available for stability
            if self.step > self.step_start_ema:
                action = self.ema_model.sample(state)
            else:
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
