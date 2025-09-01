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
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt



class MetricsLogger:
    def __init__(self, save_dir="metrics_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data storage
        self.epoch_metrics = []
        self.step_metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # For loss plotting
        self.loss_history = {
            'epochs': [],
            'losses': {},
            'rewards': []
        }
        self.best_reward = float('-inf')  # Track best reward for milestone saving
        
    def log_epoch_metrics(self, epoch, rewards, agent_losses=None):
        """Log metrics mỗi epoch"""
        if isinstance(rewards, (int, float)):
            rewards = [rewards]  # Convert single reward to list
            
        epoch_data = {
            'epoch': epoch,
            'avg_reward': np.mean(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'std_reward': np.std(rewards),
            'total_episodes': len(rewards),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add agent losses if available
        if agent_losses:
            epoch_data.update(agent_losses)
            
        self.epoch_metrics.append(epoch_data)
        
        # Update loss history for plotting
        self.loss_history['epochs'].append(epoch)
        avg_reward = np.mean(rewards)
        self.loss_history['rewards'].append(avg_reward)
        
        # Check for reward improvement for milestone saving
        reward_improved = False
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            reward_improved = True
        
        if agent_losses:
            for loss_name, loss_value in agent_losses.items():
                if loss_name not in self.loss_history['losses']:
                    self.loss_history['losses'][loss_name] = []
                self.loss_history['losses'][loss_name].append(loss_value)
                
        return reward_improved  # Return whether reward improved
                
    def save_loss_plot(self, save_interval=50):
        """Tự động lưu plot loss và reward dưới dạng PNG"""
        if len(self.loss_history['epochs']) == 0:
            return
            
        current_epoch = self.loss_history['epochs'][-1]
        
        # Create figure with subplots
        num_plots = 1 + len(self.loss_history['losses'])  # 1 for rewards + losses
        if num_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 4))
            axes = [axes]
        else:
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
            
        # Plot rewards
        axes[0].plot(self.loss_history['epochs'], self.loss_history['rewards'], 'b-', linewidth=2, label='Average Reward')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Progress - Average Reward per Episode')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot losses
        plot_idx = 1
        for loss_name, loss_values in self.loss_history['losses'].items():
            if len(loss_values) > 0:
                # Ensure same length as epochs
                epochs_for_loss = self.loss_history['epochs'][-len(loss_values):]
                axes[plot_idx].plot(epochs_for_loss, loss_values, 'r-', linewidth=2, label=loss_name)
                axes[plot_idx].set_xlabel('Episode')
                axes[plot_idx].set_ylabel('Loss')
                axes[plot_idx].set_title(f'Training Progress - {loss_name}')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].legend()
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"loss_plot.png"
        plot_path = os.path.join(self.save_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss plot saved: {plot_path}")
        
    def save_final_plot(self):
        """Lưu plot cuối cùng với tất cả dữ liệu"""
        if len(self.loss_history['epochs']) == 0:
            return
            
        # Create comprehensive final plot
        num_plots = 1 + len(self.loss_history['losses'])
        if num_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(15, 4))
            axes = [axes]
        else:
            fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
            
        # Plot rewards with moving average
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
            stats_text = f'Final Reward: {rewards[-1]:.4f}\n'
            stats_text += f'Best Reward: {max(rewards):.4f}\n'
            stats_text += f'Average: {np.mean(rewards):.4f}\n'
            stats_text += f'Std: {np.std(rewards):.4f}'
            axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot losses
        plot_idx = 1
        for loss_name, loss_values in self.loss_history['losses'].items():
            if len(loss_values) > 0:
                epochs_for_loss = epochs[-len(loss_values):]
                axes[plot_idx].plot(epochs_for_loss, loss_values, 'r-', linewidth=2, label=loss_name)
                
                # Add moving average for loss too
                if len(loss_values) > 10:
                    window_size = min(20, len(loss_values) // 5)
                    loss_moving_avg = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
                    loss_moving_epochs = epochs_for_loss[window_size-1:]
                    axes[plot_idx].plot(loss_moving_epochs, loss_moving_avg, 'g-', linewidth=2, label=f'Moving Average ({window_size})')
                
                axes[plot_idx].set_xlabel('Episode')
                axes[plot_idx].set_ylabel('Loss')
                axes[plot_idx].set_title(f'Final Training Results - {loss_name}')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].legend()
                
                # Add loss statistics
                if len(loss_values) > 0:
                    loss_stats_text = f'Final Loss: {loss_values[-1]:.6f}\n'
                    loss_stats_text += f'Min Loss: {min(loss_values):.6f}\n'
                    loss_stats_text += f'Average: {np.mean(loss_values):.6f}'
                    axes[plot_idx].text(0.02, 0.98, loss_stats_text, transform=axes[plot_idx].transAxes,
                                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        # Save final plot
        final_plot_filename = f"final_training_plot_{self.timestamp}.png"
        final_plot_path = os.path.join(self.save_dir, final_plot_filename)
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎯 Final training plot saved: {final_plot_path}")
        return final_plot_path
        
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
    
    def save_to_csv(self):
        """Save all metrics to CSV files"""
        # Save epoch metrics
        if self.epoch_metrics:
            epoch_file = os.path.join(self.save_dir, f"epoch_metrics.csv")
            with open(epoch_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.epoch_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.epoch_metrics)
            print(f"Epoch metrics saved to: {epoch_file}")
    
    def save_to_json(self):
        """Save all metrics to JSON files"""
        # Save epoch metrics
        if self.epoch_metrics:
            epoch_file = os.path.join(self.save_dir, f"epoch_metrics.json")
            with open(epoch_file, 'w') as f:
                json.dump(self._convert_for_json(self.epoch_metrics), f, indent=2)
            print(f"Epoch metrics (JSON) saved to: {epoch_file}")

hyperparameters = {
    'gail-service-env':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v1':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v3':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1}
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
    if   args.algo == 'dql':
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
    elif args.algo == 'dppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                    #   discount=args.discount,
                    #   tau=args.tau,
                    #   beta_schedule=args.beta_schedule,
                    #   n_timesteps=args.T,
                    #   lr=args.lr
                    )
    elif args.algo == 'dppo_v1':
        from agents.ppo_diffusion_v1 import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device
                      )
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        # agent = Agent(state_dim=state_dim,
        #               action_dim=action_dim,
        #               max_action=max_action,
        #               device=device,
        #               discount=args.discount,
        #               tau=args.tau,
        #               lr=args.lr,
        #               lr_decay=args.lr_decay,
        #               lr_maxt=args.num_epochs,
        #               grad_norm=args.gn,
        #               clip_ratio=0.2,
        #               value_clip_ratio=0.2,
        #               norm_adv=True,
        #               horizon_steps=1,
        #               ent_coef=0.01)

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
                      discount=args.discount,
                      tau=args.tau,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
        # agent = Agent(
        #     state_dim=state_dim,
        #     action_dim=action_dim,
        #     max_action=max_action,
        #     device=device,
        #     lr=0.0003,               
        #     noise_scale=0.1,         
        #     clip_ratio=0.1,          
        #     value_clip_ratio=0.1,    
        #     ent_coef=0.02,           
        #     norm_adv=True,
        #     discount=0.97,
        #     grad_norm=0.5
        # )
    elif args.algo == 'a2c':
        from agents.gaussian_a2c import Gaussian_A2C as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device
                      )
    elif args.algo == 'da2c':
        from agents.a2c_diffusion import Diffusion_A2C as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    writer = None 

    evaluations = []
    utils.print_banner(f"Training Start", separator="*", num_star=90)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    
    metrics_logger = MetricsLogger(save_dir=os.path.join(output_dir, "metrics"))

    episode_rewards = []  

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
                    for loss_name, loss_value in loss_info.items():
                        if loss_name not in episode_losses:
                            episode_losses[loss_name] = []
                        episode_losses[loss_name].append(loss_value)
                elif loss_info is not None:
                    if 'agent_loss' not in episode_losses:
                        episode_losses['agent_loss'] = []
                    episode_losses['agent_loss'].append(loss_info)
                    
        episode_rewards.append(episode_reward)  
        
        
        agent_episode_losses = None
        if episode_losses:
            agent_episode_losses = {}
            for loss_name, loss_values in episode_losses.items():
                if loss_values:
                    agent_episode_losses[f"avg_{loss_name}"] = np.mean(loss_values)
                    agent_episode_losses[f"std_{loss_name}"] = np.std(loss_values)
        
        reward_improved = metrics_logger.log_epoch_metrics(episode, episode_reward, agent_episode_losses)
        
        if (episode + 1) % 10 == 0:
            print(saved_info)
            metrics_logger.save_loss_plot(save_interval=50)
            
        if reward_improved and (episode + 1) % 25 == 0:
            print(f"🎯 New best reward achieved: {episode_reward:.4f} at episode {episode + 1}")
        
        np.save(os.path.join(output_dir, f"episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
        
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes, config=config, eval_env=env)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            # np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']), # These are not available in online RL
                            # np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']), # These are not available in online RL
                            episode, # Use episode number for logging
                            ])
        np.save(os.path.join(output_dir, "eval_res"), eval_res)
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.record_tabular('Avg Reward (last 10)', np.mean(episode_rewards[-10:]))
        logger.record_tabular('Avg Reward (last 50)', np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards))
        logger.record_tabular('Best Reward', np.max(episode_rewards))
        logger.dump_tabular()

        if args.save_best_model:
            agent.save_model(output_dir, episode) 

    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        top_k = min(len(scores) - 1, args.top_k) 
        where_k = scores[:, -1] == top_k 
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

    if not os.path.exists(os.path.join(output_dir, "main")):
        os.makedirs(os.path.join(output_dir, "main"))
    utils.print_banner(f"Training Complete - Saving Episode Rewards", separator="*", num_star=90)
    np.save(os.path.join(output_dir, f"main/episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
    
    utils.print_banner(f"Training Complete - Saving Final Plot", separator="*", num_star=90)
    metrics_logger.save_final_plot()
    metrics_logger.save_to_csv()
    metrics_logger.save_to_json()
    
    print(f"📊 Final training plots and metrics saved to: {metrics_logger.save_dir}")

    writer.close()

def eval_policy(policy, env_name, seed, eval_episodes=10, config = None, eval_env = None):
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
    parser.add_argument("--env_name", default="gail-service-env-v1", type=str)
    parser.add_argument("--dir", default="results", type=str)                
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
    parser.add_argument("--algo", default="dql", type=str)  # ['bc', 'ql']
    parser.add_argument("--ms", default='online', type=str, help="['online', 'offline']")
    

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
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'/{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    # env = gym.make(args.env_name)
    if args.env_name == 'gail-service-env':
        from env.gai_env_org import GAIServiceEnv, EnvConfig
    elif args.env_name == 'gail-service-env-v1':    
        from env.env import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v3':
        from env.env_v3 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    config =  EnvConfig(args.env_name)

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

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
