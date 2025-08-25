# Copyright 2024 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import os
import torch
import json
from tqdm import tqdm
from utils import utils
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from env.gai_env_org import GAIServiceEnv, EnvConfig
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('default')  # nền trắng mặc định

from utils.metrics_logger import MetricsLogger  # logger chung


hyperparameters = {
    'gail-service-env':     {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v1':  {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v3':  {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1}
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


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    if args.algo == 'dql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim, action_dim, max_action, device,
                      discount=args.discount, tau=args.tau,
                      max_q_backup=args.max_q_backup, beta_schedule=args.beta_schedule,
                      n_timesteps=args.T, eta=args.eta, lr=args.lr,
                      lr_decay=args.lr_decay, lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'dppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim, action_dim, max_action, device)
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(state_dim, action_dim, max_action, device,
                      lr=0.0003, noise_scale=0.1,
                      clip_ratio=0.1, value_clip_ratio=0.1,
                      ent_coef=0.02, norm_adv=True,
                      discount=0.97, grad_norm=0.5)
    elif args.algo == 'gdql':
        from agents.gaussian_dql import Gaussian_DQL as Agent
        agent = Agent(state_dim, action_dim, max_action, device,
                      discount=args.discount, tau=args.tau,
                      lr=args.lr, lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs, grad_norm=args.gn)
    elif args.algo == 'a2c':
        from agents.gaussian_a2c import Gaussian_A2C as Agent
        agent = Agent(state_dim, action_dim, max_action, device)
    elif args.algo == 'da2c':
        from agents.a2c_diffusion import Diffusion_A2C as Agent
        agent = Agent(state_dim, action_dim, max_action, device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    metrics_logger = MetricsLogger(save_dir=os.path.join(output_dir, "metrics"))

    episode_rewards = []
    evaluations = []

    utils.print_banner("Training Start", separator="*", num_star=90)

    for episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_losses = {}
        saved_info = None

        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            saved_info = info

            if replay_buffer.size > args.batch_size:
                loss_info = agent.train(replay_buffer, iterations=1, batch_size=args.batch_size)
                if isinstance(loss_info, dict):
                    for k, v in loss_info.items():
                        episode_losses.setdefault(k, []).append(v)
                elif loss_info is not None:
                    episode_losses.setdefault('agent_loss', []).append(loss_info)

        episode_rewards.append(episode_reward)

        # avg losses
        agent_losses = None
        if episode_losses:
            agent_losses = {f"avg_{k}": np.mean(v) for k, v in episode_losses.items()}
            agent_losses.update({f"std_{k}": np.std(v) for k, v in episode_losses.items()})

        reward_improved = metrics_logger.log_epoch_metrics(episode, episode_reward, agent_losses)

        if (episode + 1) % 10 == 0:  # save plot mỗi 10 ep
            print(saved_info)
            metrics_logger.save_loss_plot()

        if reward_improved and (episode + 1) % 25 == 0:
            print(f"🎯 New best reward: {episode_reward:.4f} at episode {episode+1}")

        np.save(os.path.join(output_dir, f"episode_rewards_{args.algo}.npy"), np.array(episode_rewards))

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed, args.eval_episodes)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, episode])
        np.save(os.path.join(output_dir, "eval"), evaluations)

        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.record_tabular('Avg Reward (last 10)', np.mean(episode_rewards[-10:]))
        logger.record_tabular('Avg Reward (last 50)', np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards))
        logger.record_tabular('Best Reward', np.max(episode_rewards))
        logger.dump_tabular()

        if args.save_best_model:
            agent.save_model(output_dir, episode)

    # Save results
    utils.print_banner("Training Complete", separator="*", num_star=90)
    np.save(os.path.join(output_dir, f"main/episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
    metrics_logger.save_final_plot()
    metrics_logger.save_to_csv()
    metrics_logger.save_to_json()
    print(f"📊 Final training plots and metrics saved to: {metrics_logger.save_dir}")


def eval_policy(policy, env_name, seed, eval_episodes=10):
    config = EnvConfig(env_name)
    eval_env = GAIServiceEnv(config)
    np.random.seed(seed + 100)

    scores = []
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)
    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f}")
    return avg_reward, std_reward, avg_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='exp_1', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--env_name", default="gail-service-env-v1", type=str)
    parser.add_argument("--dir", default="tests", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_episodes", default=500, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='linear', type=str)
    parser.add_argument("--algo", default="dql", type=str)
    parser.add_argument("--ms", default='online', type=str)
    args = parser.parse_args()

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    hp = hyperparameters[args.env_name]
    args.num_epochs = hp['num_epochs']
    args.eval_freq = hp['eval_freq']
    args.eval_episodes = 100
    args.lr = hp['lr']; args.eta = hp['eta']
    args.max_q_backup = hp['max_q_backup']; args.reward_tune = hp['reward_tune']
    args.gn = hp['gn']; args.top_k = hp['top_k']

    file_name = f"{args.env_name}|{args.exp}/diffusion-{args.algo}|T-{args.T}|ms-{args.ms}/{args.seed}"
    results_dir = os.path.join(args.output_dir, file_name)
    os.makedirs(results_dir, exist_ok=True)
    utils.print_banner(f"Saving location: {results_dir}")
    variant = vars(args); variant.update(version="Diffusion-Policies-RL")

    config = EnvConfig(args.env_name)
    env = GAIServiceEnv(config)
    env.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
