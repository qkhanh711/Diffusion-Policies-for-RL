# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
import os
import torch
# from utils import utils
from utils.utils import print_banner
from utils.logger import logger, setup_logger
# from jlgo.uav_env import UAVGenAIEnv
from jlgo.uav_env_JLGO import UAVGenAIEnv
from tqdm import tqdm

hyperparameters = {
    'uav-genai-env': {
        'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
        'eval_freq': 50, 'num_epochs': 10000, 'num_episodes_per_epoch': 5,
        'gn': 5.0, 'top_k': 1
    },
}

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=10000, device='cpu'):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self._size = 0
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

    def add(self, state, action, next_state, reward, done):
        idx = self.ptr % self.max_size
        self.state[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.action[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.next_state[idx] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.reward[idx] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.done[idx] = torch.tensor([done], dtype=torch.float32, device=self.device)
        self.ptr += 1
        self._size = min(self._size + 1, self.max_size)

    @property
    def size(self):
        return self._size

    def sample(self, batch_size):
        # Randomly sample batch_size transitions
        indices = np.random.randint(0, self._size, size=batch_size)
        
        return (
            self.state[indices],
            self.action[indices], 
            self.next_state[indices],
            self.reward[indices],
            1 - self.done[indices]  # not_done = 1 - done
        )

def get_uav_config():
    return {
        "num_users": 5,
        "Dmax": 20,
        "tau": 30.0,  # Increased timeout to match config.yaml
        "area": [-500, 500, -500, 500],
        "z_range": [100, 1000],
        "BS_position": [0, 0, 50],
        "phi": 1.5e-9,
        "f_inf_BS": 1.0e9,
        "Cin_BS": 0.03125,
        "lambda_Q": 0.5,
        "lambda_L": 0.5,
        "lambda_E": 1.0,
        "psi": 10.0,
        "Qreq": [30, 30, 30, 30, 30],
        "T": 10,
        "communication": {
            "tau_slot_duration": 30.0  # Increased timeout for transmission
        }
    }

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    if args.algo == 'dql':
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
    elif args.algo == 'ppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim, 
                      max_action=max_action,
                      device=device,
                      lr=0.0005)
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=0.0003,                # giảm learning rate
            noise_scale=0.1,          # giảm noise
            clip_ratio=0.1,           # giảm clip ratio
            value_clip_ratio=0.1,     # giảm value clip ratio
            ent_coef=0.02,            # tăng entropy coef
            norm_adv=True,
            discount=0.97,
            grad_norm=0.5
        )
    elif args.algo == 'gdql':
        from agents.gaussian_dql import Gaussian_DQL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      lr=0.001,
                      noise_scale=0.3,
                      noise_type='gaussian',
                      epsilon=0.01
                      )
    elif args.algo == 'a2c':
        from agents.a2c_agent import A2C_Agent as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=0.0009,
            gamma=0.99,
            value_coef=0.45,
            ent_coef=0.01,
            grad_norm=1.0
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    replay_buffer = ReplayBuffer(state_dim, action_dim, 1_000_000, device)
    rewards = []

    print_banner("Training Start", separator="*", num_star=90)
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        total_reward = 0
        for _ in range(args.num_episodes_per_epoch):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            done = False
            ep_reward = 0
            while not done:
                action = agent.sample_action(state)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()[0]
                
                # Handle Dict action space
                if hasattr(env.action_space, 'spaces'):
                    # Convert flat action to Dict format
                    uav_movement_dim = env.action_space['uav_movement'].shape[0]
                    denoising_dim = env.action_space['denoising_steps'].shape[0]
                    
                    action_dict = {
                        "uav_movement": action[:uav_movement_dim],
                        "denoising_steps": action[uav_movement_dim:uav_movement_dim + denoising_dim].astype(int)
                    }
                    env_action = action_dict
                else:
                    env_action = action
                
                step_result = env.step(env_action)
                if len(step_result) == 5:  # Gym>=0.26
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym cũ
                    next_state, reward, done, info = step_result
                
                # Check for NaN or inf values in reward and states
                if np.isnan(reward) or np.isinf(reward):
                    print(f"Warning: reward is {reward}, skipping this transition")
                    continue
                    
                if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                    print(f"Warning: next_state contains NaN/inf, skipping this transition")
                    continue
                
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                ep_reward += reward
                if replay_buffer.size > args.batch_size:
                    loss = agent.train(replay_buffer, iterations=1, batch_size=args.batch_size)
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
    parser.add_argument("--algo", default="bc", type=str, choices=['bc', 'ql', 'ppo', 'dql', 'gppo', 'gdql', 'diffppo', 'a2c'])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Reset environment first to initialize all state variables
    env.reset()
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Use observation space dimension for consistency
    state_dim = env.observation_space.shape[0]
    print(f"State dimension: {state_dim}")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Handle Dict action space - get total action dimension
    if hasattr(env.action_space, 'spaces'):
        # Dict action space
        action_dim = env.action_space['uav_movement'].shape[0] + env.action_space['denoising_steps'].shape[0]
    else:
        # Simple Box action space
        action_dim = env.action_space.shape[0]
    
    max_action = 1.0  # hoặc thiết lập phù hợp nếu có action_space

    variant = vars(args)
    variant.update(version="Diffusion-Policies-RL", state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    print_banner(f"Env: uav-genai-env, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
