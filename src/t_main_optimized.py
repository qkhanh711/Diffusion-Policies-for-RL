# Copyright 2024 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json
from tqdm import tqdm
from utils import utils
from utils.logger import logger, setup_logger
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

class MetricsLogger:
    def __init__(self, save_dir="metrics_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data storage
        self.epoch_metrics = []
        self.env_metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # For loss plotting
        self.loss_history = {
            'epochs': [],
            'losses': {},
            'rewards': []
        }
        self.best_reward = float('-inf')
        
        # For GIF animations
        self.gif_episodes_data = []
        self.max_gif_episodes = 5
        
    def start_episode_recording(self):
        """Start recording step-by-step data for current episode"""
        self.current_episode_steps = []
        
    def log_step_data(self, step, state, action, reward, info, config):
        """Log data for each step within an episode"""
        try:
            num_users = config["num_users"]
            step_data = {
                'step': step,
                'reward': reward,
                'cumulative_reward': getattr(self, 'episode_cumulative_reward', 0) + reward,
                'users_data': [],
                'system_totals': {
                    'total_served': info.get('total_served', 0),
                    'total_latency': info.get('total_latency', 0),
                    'total_flops': info.get('total_flops', 0),
                    'total_mem': info.get('total_mem', 0),
                    'mean_qos': info.get('mean_qos', 0),
                    'bonus': info.get('bonus', 0),
                    'penalty': info.get('penalty', 0),
                    'denoise_steps': info.get('denoise_steps', [0]*num_users)
                }
            }
            
            # Update cumulative reward
            self.episode_cumulative_reward = step_data['cumulative_reward']
            
            # Extract user data
            for i in range(num_users):
                user_state_start = i * 6
                raw_serve_decision = float(action[2*i])
                raw_denoise_action = float(action[2*i + 1])
                
                # Normalize actions
                normalized_serve = (raw_serve_decision + 1.0) / 2.0 if raw_serve_decision < 0 or raw_serve_decision > 1 else raw_serve_decision
                normalized_denoise = (raw_denoise_action + 1.0) / 2.0 if raw_denoise_action < 0 or raw_denoise_action > 1 else raw_denoise_action
                
                normalized_serve = np.clip(normalized_serve, 0.0, 1.0)
                normalized_denoise = np.clip(normalized_denoise, 0.0, 1.0)
                
                scaled_denoise_steps = int(1 + normalized_denoise * (config["max_denoise_steps"] - 1))
                scaled_denoise_steps = np.clip(scaled_denoise_steps, 1, config["max_denoise_steps"])
                
                user_data = {
                    'user_id': i,
                    'position': {
                        'x': float(state[user_state_start]),
                        'y': float(state[user_state_start + 1])
                    },
                    'image_size': float(state[user_state_start + 2]),
                    'prompt_size': float(state[user_state_start + 3]),
                    'direction': float(state[user_state_start + 4]),
                    'qos_required': float(state[user_state_start + 5]),
                    'action': {
                        'serve_decision': normalized_serve,
                        'serve_binary': 1 if normalized_serve >= 0.5 else 0,
                        'denoise_steps': scaled_denoise_steps,
                    }
                }
                step_data['users_data'].append(user_data)
            
            self.current_episode_steps.append(step_data)
            
        except Exception as e:
            print(f"Warning: Failed to log step data: {e}")

    def end_episode_recording(self, episode_num, should_create_gif=False):
        """End episode recording and optionally prepare for GIF creation"""
        if hasattr(self, 'current_episode_steps') and self.current_episode_steps:
            episode_data = {
                'episode': episode_num,
                'steps': self.current_episode_steps.copy(),
                'total_steps': len(self.current_episode_steps),
                'final_reward': self.current_episode_steps[-1]['cumulative_reward'] if self.current_episode_steps else 0
            }
            
            if should_create_gif and len(self.gif_episodes_data) < self.max_gif_episodes:
                self.gif_episodes_data.append(episode_data)
                
            self.current_episode_steps = []
            self.episode_cumulative_reward = 0

    def log_env_metrics(self, episode, env_info, action, state, config):
        """Log environment-specific metrics"""
        try:
            num_users = config["num_users"]
            
            env_data = {
                'episode': episode,
                'timestamp': datetime.now().isoformat(),
                'system_constraints': {
                    'Gmax': float(config["Gmax"]),
                    'Mmax': float(config["Mmax"]),
                    'sys_tau': float(config["sys_tau"]),
                    'max_denoise_steps': int(config["max_denoise_steps"])
                },
                'users_data': [],
                'system_totals': {
                    'total_served': int(env_info.get('total_served', 0)),
                    'total_latency': float(env_info.get('total_latency', 0)),
                    'total_flops': float(env_info.get('total_flops', 0)),
                    'total_mem': float(env_info.get('total_mem', 0)),
                    'bonus': float(env_info.get('bonus', 0)),
                    'penalty': float(env_info.get('penalty', 0)),
                    'mean_qos': float(env_info.get('mean_qos', 0)),
                    'denoise_steps': [int(ds) for ds in env_info.get('denoise_steps', [0]*num_users)] if isinstance(env_info.get('denoise_steps', []), (list, np.ndarray)) else [0]*num_users
                },
                'constraints_status': {
                    'latency_ok': bool(env_info.get('total_latency', 0) <= config["sys_tau"] * num_users),
                    'flops_ok': bool(env_info.get('total_flops', 0) <= config["Gmax"]),
                    'mem_ok': bool(env_info.get('total_mem', 0) <= config["Mmax"])
                }
            }
            
            # Extract user-specific information
            for i in range(num_users):
                user_state_start = i * 6
                try:
                    raw_serve_decision = float(action[2*i])
                    raw_denoise_action = float(action[2*i + 1])
                except (IndexError, TypeError):
                    raw_serve_decision = 0.0
                    raw_denoise_action = 0.0
                
                normalized_serve = (raw_serve_decision + 1.0) / 2.0 if raw_serve_decision < 0 or raw_serve_decision > 1 else raw_serve_decision
                normalized_denoise = (raw_denoise_action + 1.0) / 2.0 if raw_denoise_action < 0 or raw_denoise_action > 1 else raw_denoise_action
                
                normalized_serve = np.clip(normalized_serve, 0.0, 1.0)
                normalized_denoise = np.clip(normalized_denoise, 0.0, 1.0)
                
                scaled_denoise_steps = int(1 + normalized_denoise * (config["max_denoise_steps"] - 1))
                scaled_denoise_steps = np.clip(scaled_denoise_steps, 1, config["max_denoise_steps"])
                
                try:
                    user_data = {
                        'user_id': i,
                        'position': {
                            'x': float(state[user_state_start]),
                            'y': float(state[user_state_start + 1])
                        },
                        'image_size': float(state[user_state_start + 2]),
                        'prompt_size': float(state[user_state_start + 3]),
                        'direction': float(state[user_state_start + 4]),
                        'qos_required': float(state[user_state_start + 5]),
                        'action': {
                            'serve_decision': float(normalized_serve),
                            'serve_binary': int(1 if normalized_serve >= 0.5 else 0),
                            'denoise_steps': int(scaled_denoise_steps),
                        }
                    }
                    env_data['users_data'].append(user_data)
                except IndexError:
                    print(f"Warning: State index out of bounds for user {i}")
            
            self.env_metrics.append(env_data)
            
        except Exception as e:
            print(f"Warning: Failed to log environment metrics: {e}")

    def log_epoch_metrics(self, epoch, rewards, agent_losses=None):
        """Log metrics per epoch"""
        if isinstance(rewards, (int, float)):
            rewards = [rewards]
            
        epoch_data = {
            'epoch': epoch,
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards),
            'total_episodes': len(rewards),
            'timestamp': datetime.now().isoformat()
        }
        
        if agent_losses:
            epoch_data.update(agent_losses)
            
        self.epoch_metrics.append(epoch_data)
        
        # Update loss history for plotting
        self.loss_history['epochs'].append(epoch)
        avg_reward = np.mean(rewards)
        self.loss_history['rewards'].append(avg_reward)
        
        # Check for reward improvement
        reward_improved = False
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            reward_improved = True
        
        if agent_losses:
            for loss_name, loss_value in agent_losses.items():
                if loss_name not in self.loss_history['losses']:
                    self.loss_history['losses'][loss_name] = []
                self.loss_history['losses'][loss_name].append(loss_value)
                
        return reward_improved

    def save_final_plot(self):
        """Save final comprehensive plot"""
        if len(self.loss_history['epochs']) == 0:
            return
            
        num_plots = 1 + len(self.loss_history['losses'])
        if num_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(15, 4))
            axes = [axes]
        else:
            fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
            
        epochs = self.loss_history['epochs']
        rewards = self.loss_history['rewards']
        
        axes[0].plot(epochs, rewards, 'b-', linewidth=1, alpha=0.7, label='Average Reward')
        
        # Add moving average if enough data
        if len(rewards) > 10:
            window_size = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_epochs = epochs[window_size-1:]
            axes[0].plot(moving_epochs, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
            
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Final Training Results - Reward Progress')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add statistics text
        if len(rewards) > 0:
            stats_text = f'Final: {rewards[-1]:.4f}\nBest: {max(rewards):.4f}\nAvg: {np.mean(rewards):.4f}'
            axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot losses
        plot_idx = 1
        for loss_name, loss_values in self.loss_history['losses'].items():
            if len(loss_values) > 0 and plot_idx < len(axes):
                epochs_for_loss = self.loss_history['epochs'][-len(loss_values):]
                axes[plot_idx].plot(epochs_for_loss, loss_values, 'r-', linewidth=2, label=loss_name)
                axes[plot_idx].set_xlabel('Episode')
                axes[plot_idx].set_ylabel('Loss')
                axes[plot_idx].set_title(f'Training Progress - {loss_name}')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].legend()
                plot_idx += 1
        
        plt.tight_layout()
        
        final_plot_path = os.path.join(self.save_dir, "final_training_plot.png")
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎯 Final training plot saved: {final_plot_path}")
        return final_plot_path

    def save_env_metrics_plot(self):
        """Create simplified environment metrics plot"""
        if len(self.env_metrics) == 0:
            print("No environment metrics to plot")
            return
            
        try:
            episodes = [m['episode'] for m in self.env_metrics]
            served_users = [m['system_totals'].get('total_served', 0) for m in self.env_metrics]
            
            # Calculate average denoise steps
            avg_denoise_steps = []
            for m in self.env_metrics:
                denoise_steps = m['system_totals'].get('denoise_steps', [])
                if isinstance(denoise_steps, (list, np.ndarray)) and denoise_steps:
                    avg_steps = np.mean([s for s in denoise_steps if s > 0])
                    avg_denoise_steps.append(avg_steps)
                else:
                    avg_denoise_steps.append(0)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Environment Metrics Over Episodes', fontsize=16)
            
            # Plot 1: Served Users
            axes[0, 0].plot(episodes, served_users, 'g-', linewidth=2)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Users Served')
            axes[0, 0].set_title('Users Served per Episode')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Average Denoise Steps
            axes[0, 1].plot(episodes, avg_denoise_steps, 'orange', linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Denoise Steps')
            axes[0, 1].set_title('Average Denoise Steps')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Resource Usage
            total_latency = [m['system_totals'].get('total_latency', 0) for m in self.env_metrics]
            total_flops = [m['system_totals'].get('total_flops', 0) for m in self.env_metrics]
            
            if any(total_latency):
                axes[1, 0].plot(episodes, total_latency, 'r-', linewidth=2, label='Latency')
            if any(total_flops):
                axes[1, 0].plot(episodes, total_flops, 'b-', linewidth=2, label='FLOPS')
            
            if any(total_latency) or any(total_flops):
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Resource Usage')
                axes[1, 0].set_title('System Resources')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No resource data', ha='center', va='center', transform=axes[1, 0].transAxes)

            # Plot 4: Rewards/Penalties
            bonuses = [m['system_totals'].get('bonus', 0) for m in self.env_metrics]
            penalties = [m['system_totals'].get('penalty', 0) for m in self.env_metrics]
            
            if any(bonuses):
                axes[1, 1].plot(episodes, bonuses, 'g-', linewidth=2, label='Bonus')
            if any(penalties):
                axes[1, 1].plot(episodes, penalties, 'r-', linewidth=2, label='Penalty')
            
            if any(bonuses) or any(penalties):
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].set_title('Rewards and Penalties')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'No reward data', ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()
            
            env_plot_path = os.path.join(self.save_dir, "environment_metrics.png")
            plt.savefig(env_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Environment metrics plot saved: {env_plot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create environment metrics plot: {e}")

    def save_user_positions_plot(self):
        """Create simple user positions plot"""
        if len(self.env_metrics) == 0:
            return
            
        try:
            last_episode_data = self.env_metrics[-1]
            users_data = last_episode_data['users_data']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            for user in users_data:
                x, y = user['position']['x'], user['position']['y']
                serve_status = user['action']['serve_binary']
                denoise_steps = user['action']['denoise_steps']
                
                color = 'green' if serve_status else 'red'
                size = max(50, denoise_steps * 10) if serve_status else 30
                
                ax.scatter(x, y, c=color, s=size, alpha=0.7, 
                          label=f"User {user['user_id']}: {'Served' if serve_status else 'Not Served'}")
            
            # Service provider position
            ax.scatter(0.0, 0.0, c='blue', s=200, marker='s', 
                      label='Service Provider', edgecolors='black', linewidth=2)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'User Positions (Episode {last_episode_data["episode"]})')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            pos_plot_path = os.path.join(self.save_dir, "user_positions.png")
            plt.savefig(pos_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📍 User positions plot saved: {pos_plot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create user positions plot: {e}")

    def _convert_for_json(self, data):
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: self._convert_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

    def save_to_json(self):
        """Save metrics to JSON files"""
        if self.epoch_metrics:
            epoch_file = os.path.join(self.save_dir, "epoch_metrics.json")
            with open(epoch_file, 'w') as f:
                json.dump(self._convert_for_json(self.epoch_metrics), f, indent=2)
            print(f"Epoch metrics (JSON) saved to: {epoch_file}")
        
        if self.env_metrics:
            env_file = os.path.join(self.save_dir, "environment_metrics.json")
            with open(env_file, 'w') as f:
                json.dump(self._convert_for_json(self.env_metrics), f, indent=2)
            print(f"Environment metrics (JSON) saved to: {env_file}")


# Hyperparameters dictionary
hyperparameters = {
    'gail-service-env':         {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v1':      {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v3-org':  {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 25, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v4':      {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1},
    'gail-service-env-v5':      {'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0, 'top_k': 1}
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


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args, config):
    # Agent initialization based on algorithm
    if args.algo == 'dql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device,
                      discount=args.discount, tau=args.tau, max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule, n_timesteps=args.T, eta=args.eta,
                      lr=args.lr, lr_decay=args.lr_decay, lr_maxt=args.num_epochs, grad_norm=args.gn)
    elif args.algo == 'dppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    elif args.algo == 'dppo_v1':
        from agents.ppo_diffusion_v1 import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device,
                      lr=0.0003, noise_scale=0.1, clip_ratio=0.1, value_clip_ratio=0.1,
                      ent_coef=0.02, norm_adv=True, discount=0.97, grad_norm=0.5)
    elif args.algo == 'gdql':
        from agents.gaussian_dql import Gaussian_DQL as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device,
                      discount=args.discount, tau=args.tau, lr=args.lr, lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs, grad_norm=args.gn)
    elif args.algo == 'a2c':
        from agents.gaussian_a2c import Gaussian_A2C as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    elif args.algo == 'da2c':
        from agents.a2c_diffusion import Diffusion_A2C as Agent
        agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    evaluations = []
    utils.print_banner(f"Training Start", separator="*", num_star=90)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    metrics_logger = MetricsLogger(save_dir=os.path.join(output_dir, "metrics"))

    episode_rewards = []
    current_state = env.reset()

    for episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        current_state = state
        done = False
        episode_reward = 0
        episode_losses = {}
        saved_info = None
        saved_action = None

        should_record_gif = (episode % args.num_episodes == 0)
        if should_record_gif:
            metrics_logger.start_episode_recording()
        
        step_count = 0
        while not done:
            action = agent.sample_action(state)
            saved_action = action
            next_state, reward, done, info = env.step(action)
            
            if args.print_logger and step_count == 0:  # Print only first step to reduce output
                print(f"Episode {episode}, Step {step_count}: Reward = {reward:.4f}")
            
            if should_record_gif:
                metrics_logger.log_step_data(step_count, state, action, reward, info, config)
            
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            current_state = state
            episode_reward += reward
            saved_info = info
            step_count += 1

            if replay_buffer.size > args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                losses = agent.train(batch)
                if losses:
                    for loss_name, loss_value in losses.items():
                        if loss_name not in episode_losses:
                            episode_losses[loss_name] = []
                        episode_losses[loss_name].append(loss_value)
                    
        if should_record_gif:
            metrics_logger.end_episode_recording(episode, should_create_gif=True)
        
        if args.print_logger and saved_info is not None:
            print(f"Episode {episode} Summary: Reward={episode_reward:.4f}, "
                  f"Served={saved_info.get('total_served', 0)}/{config.get('num_users', 0)}")
                    
        episode_rewards.append(episode_reward)
        
        if saved_info is not None and saved_action is not None:
            metrics_logger.log_env_metrics(episode, saved_info, saved_action, current_state, config)
        
        agent_episode_losses = None
        if episode_losses:
            agent_episode_losses = {}
            for loss_name, loss_values in episode_losses.items():
                agent_episode_losses[loss_name] = np.mean(loss_values)
        
        reward_improved = metrics_logger.log_epoch_metrics(episode, episode_reward, agent_episode_losses)

        np.save(os.path.join(output_dir, f"episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
        
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(
            agent, args.env_name, args.seed, eval_episodes=args.eval_episodes, config=config, eval_env=env)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, episode])
        
        if episode % args.eval_freq == 0:
            if reward_improved:
                print(f"🎯 New best reward at episode {episode}: {reward_improved}")
            metrics_logger.save_final_plot()
            metrics_logger.save_env_metrics_plot()
            metrics_logger.save_user_positions_plot()
            metrics_logger.save_to_json()
            
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.record_tabular('Avg Reward (last 10)', np.mean(episode_rewards[-10:]))
        logger.record_tabular('Best Reward', np.max(episode_rewards))
        logger.dump_tabular()

        if args.save_best_model:
            agent.save_model(output_dir, episode)

    # Save final results
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {
            'model selection': args.ms, 
            'epoch': scores[best_id, -1],
            'best normalized score avg': scores[best_id, 2],
            'best normalized score std': scores[best_id, 3],
            'best raw score avg': scores[best_id, 0],
            'best raw score std': scores[best_id, 1]
        }
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        top_k = min(len(scores) - 1, args.top_k)
        where_k = scores[:, -1] == top_k
        best_res = {
            'model selection': args.ms, 
            'epoch': scores[where_k][0][-1],
            'best normalized score avg': scores[where_k][0][2],
            'best normalized score std': scores[where_k][0][3],
            'best raw score avg': scores[where_k][0][0],
            'best raw score std': scores[where_k][0][1]
        }
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    if not os.path.exists(os.path.join(output_dir, "main")):
        os.makedirs(os.path.join(output_dir, "main"))
    
    utils.print_banner(f"Training Complete - Saving Final Results", separator="*", num_star=90)
    np.save(os.path.join(output_dir, f"main/episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
    
    metrics_logger.save_final_plot()
    metrics_logger.save_env_metrics_plot()
    metrics_logger.save_user_positions_plot()
    metrics_logger.save_to_json()
    
    print(f"📊 Final training results saved to: {metrics_logger.save_dir}")


def eval_policy(policy, env_name, seed, eval_episodes=10, config=None, eval_env=None):
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
    avg_norm_score = avg_reward
    std_norm_score = std_reward

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='exp_1', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument("--env_name", default="gail-service-env-v3", type=str)
    parser.add_argument("--dir", default="t_results", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_episodes", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='linear', type=str)
    parser.add_argument("--algo", default="dql", type=str)
    parser.add_argument("--ms", default='online', type=str, help="['online', 'offline']")
    parser.add_argument("--n_users", default=10, type=int, help="Number of users in the environment")
    parser.add_argument("--print_logger", action='store_true', help="Print detailed resource usage to terminal")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 100 if 'v2' in args.env_name else 100

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}/diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: 
        file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'
    if args.ms == 'offline': 
        file_name += f'|k-{args.top_k}'
    file_name += f'/{args.n_users}/{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    # Environment setup
    if args.env_name == 'gail-service-env':
        from env.gai_env_org import GAIServiceEnv, EnvConfig
    elif args.env_name == 'gail-service-env-v1':    
        from env.env import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v5':
        from env.env_v5_org import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v4':
        from env.env_v4 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v3-org':
        from env.env_v3_org import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig

    config = EnvConfig(args.env_name)
    config["num_users"] = args.n_users

    print(f"Training config: {config}")
    env = GAIServiceEnv(config)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args, config)
