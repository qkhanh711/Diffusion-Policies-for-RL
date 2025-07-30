# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

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
            nn.Mish(),
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
                 lr=3e-4,
                 ema_decay=0.995,
                 lr_decay=False,
                 lr_maxt=1000):

        self.device = device
        self.actor = Diffusion(state_dim=state_dim,
                               action_dim=action_dim,
                               model=MLP(state_dim, action_dim, device),
                               max_action=max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optim, T_max=lr_maxt)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optim, T_max=lr_maxt)

    def compute_gae(self, rewards, values, dones, next_value):
        values = np.append(values, next_value)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(self, states, actions, log_probs_old, returns, advantages, epochs=1, batch_size=100):
        states = states.to(self.device)
        actions = actions.to(self.device)
        log_probs_old = log_probs_old.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        metric = {'ppo_loss': [], 'value_loss': []}

        dataset_size = len(states)
        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_log_probs_old = log_probs_old[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                new_actions = self.actor(batch_states)
                new_log_probs = -self.actor.loss(batch_actions, batch_states)

                ratio = torch.exp(new_log_probs - batch_log_probs_old)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_pred = self.critic(batch_states).squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns)
                self.actor_optim.zero_grad()
                policy_loss.backward()
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

                metric['ppo_loss'].append(policy_loss.item())
                metric['value_loss'].append(value_loss.item())
        return metric
    
    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        for i in range(iterations):
            states, actions, log_probs_old, returns, advantages = replay_buffer.sample(batch_size)
            metric = self.ppo_update(states, actions, log_probs_old, returns, advantages)    
            if log_writer is not None:
                log_writer.add_scalar('PPO Loss', metric['ppo_loss'], i)
                log_writer.add_scalar('Value Loss', metric['value_loss'], i)
            return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path + "/ppo_actor.pth")
        torch.save(self.critic.state_dict(), path + "/ppo_critic.pth")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + "/ppo_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "/ppo_critic.pth"))
