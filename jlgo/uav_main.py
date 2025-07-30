# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import os
import torch
# from utils import utils
from utils.utils import print_banner
from utils.logger import logger, setup_logger
from uav_env import UAVGenAIEnv

hyperparameters = {
    'uav-genai-env': {
        'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
        'eval_freq': 50, 'num_epochs': 2000, 'num_episodes_per_epoch': 10,
        'gn': 5.0, 'top_k': 1
    },
}

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

def get_uav_config():
    return {
        "num_users": 5, "Dmax": 20, "tau": 3.0, "area": [-500, 500, -500, 500],
        "z_range": [100, 1000], "BS_position": [0, 0, 50], "phi": 1.5e-9,
        "f_inf_BS": 1.0e9, "Cin_BS": 0.03125, "lambda_Q": 0.5,
        "lambda_L": 0.5, "lambda_E": 1.0, "psi": 10.0, "Qreq": [30]*5, "T": 10
    }

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    if args.algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)
    elif args.algo == 'ppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim, action_dim, max_action, device, args.discount, args.tau,
                      args.beta_schedule, args.T, args.lr, clip_ratio=0.2,
                      value_clip_ratio=0.2, norm_adv=True)
    elif args.algo == 'dql':
        from agents.dql_diffusion import Diffusion_DQL as Agent
        # agent = Agent(state_dim, action_dim, max_action, device, args.discount, args.tau,
        #               args.beta_schedule, args.T, args.lr)
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      eta=args.eta,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'gaussian_ppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn,
                      clip_ratio=0.2,
                      value_clip_ratio=0.2,
                      norm_adv=True,
                      horizon_steps=1,
                      ent_coef=0.01)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    replay_buffer = ReplayBuffer(state_dim, action_dim, 1_000_000, device)
    rewards = []

    print_banner("Training Start", separator="*", num_star=90)
    for epoch in range(args.num_epochs):
        total_reward = 0
        for _ in range(args.num_episodes_per_epoch):
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = agent.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                ep_reward += reward
                if replay_buffer.size > args.batch_size:
                    agent.train(replay_buffer, iterations=1, batch_size=args.batch_size)
            total_reward += ep_reward
        avg_reward = total_reward / args.num_episodes_per_epoch
        rewards.append(avg_reward)
        if (epoch + 1) % 10 == 0:
            logger.record_tabular('Epoch', epoch + 1)
            logger.record_tabular('Epoch Reward', avg_reward)
            logger.record_tabular('Avg Reward (last 10)', np.mean(rewards[-10:]))
            logger.dump_tabular()

    np.save(os.path.join(output_dir, "epoch_rewards.npy"), np.array(rewards))
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='exp_1', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--dir", default="results", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=2000, type=int)
    parser.add_argument("--num_episodes_per_epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    # Algorithm choice
    parser.add_argument("--algo", default="bc", type=str, choices=['bc', 'ql', 'ppo', 'dql', 'gaussian_ppo'])

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    hp = hyperparameters['uav-genai-env']
    args.num_epochs = hp['num_epochs']
    args.num_episodes_per_epoch = hp['num_episodes_per_epoch']
    args.lr = hp['lr']
    args.eta = hp['eta']
    args.max_q_backup = hp['max_q_backup']
    args.gn = hp['gn']
    args.top_k = hp['top_k']

    file_name = f"uav-genai-env|{args.exp}|diffusion-{args.algo}|T-{args.T}|{args.seed}"
    results_dir = os.path.join(args.dir, file_name)
    os.makedirs(results_dir, exist_ok=True)
    print_banner(f"Saving location: {results_dir}")

    config = get_uav_config()
    env = UAVGenAIEnv(config)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant = vars(args)
    variant.update(version="Diffusion-Policies-RL", state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    print_banner(f"Env: uav-genai-env, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
