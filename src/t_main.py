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
import matplotlib.animation as animation
from matplotlib.patches import Circle
import imageio



class MetricsLogger:
    def __init__(self, save_dir="metrics_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data storage
        self.epoch_metrics = []
        self.step_metrics = []
        self.env_metrics = []
        self.episode_steps_data = []  # New: Store step-by-step data for GIF
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # For loss plotting
        self.loss_history = {
            'epochs': [],
            'losses': {},
            'rewards': []
        }
        self.best_reward = float('-inf')
        
        # For GIF animations
        self.gif_episodes_data = []  # Store selected episodes for GIF creation
        self.max_gif_episodes = 5    # Limit number of episodes to create GIFs for
        
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
                    'total_served': info.get('served', 0),
                    'total_latency': info.get('total_latency', 0),
                    'total_flops': info.get('total_flops', 0),
                    'total_mem': info.get('total_mem', 0),
                    'mean_qos': info.get('mean_qos', 0),
                    'bonus': info.get('bonus', 0),
                    'penalty': info.get('penalty', 0),
                    'penalty_qos': info.get('penalty_qos', 0),
                    'latency': info.get('latency', 0),
                    'flops': info.get('flops', 0),
                    'mem': info.get('mem', 0),
                    'price': info.get('price', 0),
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
                
                # Normalize actions if they are outside [0,1] range
                # Assuming action space is [-1, 1], normalize to [0, 1]
                normalized_serve = (raw_serve_decision + 1.0) / 2.0 if raw_serve_decision < 0 or raw_serve_decision > 1 else raw_serve_decision
                normalized_denoise = (raw_denoise_action + 1.0) / 2.0 if raw_denoise_action < 0 or raw_denoise_action > 1 else raw_denoise_action
                
                # Clip to [0, 1] to be safe
                normalized_serve = np.clip(normalized_serve, 0.0, 1.0)
                normalized_denoise = np.clip(normalized_denoise, 0.0, 1.0)
                
                # Scale denoise action from [0,1] to [1, max_denoise_steps]
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
                        'raw_serve_action': raw_serve_decision,  # Keep raw for debugging
                        'raw_denoise_action': raw_denoise_action,  # Keep raw for debugging
                        'normalized_denoise': normalized_denoise  # Keep normalized for debugging
                    }
                }
                step_data['users_data'].append(user_data)
            
            self.current_episode_steps.append(step_data)
            
        except Exception as e:
            print(f"Warning: Failed to log step data: {e}")
            print(f"Debug - action shape: {action.shape if hasattr(action, 'shape') else 'no shape'}")
            print(f"Debug - action values: {action}")
            print(f"Debug - action range: min={np.min(action)}, max={np.max(action)}")

    def end_episode_recording(self, episode_num, should_create_gif=False):
        """End episode recording and optionally prepare for GIF creation"""
        if hasattr(self, 'current_episode_steps') and self.current_episode_steps:
            episode_data = {
                'episode': episode_num,
                'steps': self.current_episode_steps.copy(),
                'total_steps': len(self.current_episode_steps),
                'final_reward': self.current_episode_steps[-1]['cumulative_reward'] if self.current_episode_steps else 0
            }
            
            # Store for GIF creation if needed
            if should_create_gif and len(self.gif_episodes_data) < self.max_gif_episodes:
                self.gif_episodes_data.append(episode_data)
                
            # Reset for next episode
            self.current_episode_steps = []
            self.episode_cumulative_reward = 0
            
    def create_user_movement_gif(self, episode_data, filename_suffix=""):
        """Create GIF showing user movements and service decisions throughout an episode"""
        try:
            steps = episode_data['steps']
            if not steps:
                return
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Set up the position plot
            # ax1.set_xlim(-1.2, 1.2)
            # ax1.set_ylim(-1.2, 1.2)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title(f'User Movements - Episode {episode_data["episode"]}')
            ax1.grid(True, alpha=0.3)
            
            # Service provider position
            sp_circle = Circle((0, 0), 0.1, color='blue', alpha=0.8, label='Service Provider')
            ax1.add_patch(sp_circle)
            
            # Set up metrics plot
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Value')
            ax2.set_title('Metrics Over Time')
            ax2.grid(True, alpha=0.3)
            
            def animate(frame):
                ax1.clear()
                ax2.clear()
                
                # Current step data
                current_step = steps[frame]
                
                # Redraw service provider
                ax1.add_patch(Circle((0, 0), 0.1, color='blue', alpha=0.8))
                ax1.scatter(0, 0, c='blue', s=200, marker='s', label='Service Provider', 
                           edgecolors='black', linewidth=2)
                
                # Draw users
                for user in current_step['users_data']:
                    x, y = user['position']['x'], user['position']['y']
                    serve_status = user['action']['serve_binary']
                    denoise_steps = user['action']['denoise_steps']
                    
                    color = 'green' if serve_status else 'red'
                    size = max(100, denoise_steps * 20) if serve_status else 50
                    alpha = 0.8 if serve_status else 0.5
                    
                    ax1.scatter(x, y, c=color, s=size, alpha=alpha, 
                              label=f"User {user['user_id']}: {'Served' if serve_status else 'Not Served'}")
                    
                    # Draw connection line if served
                    if serve_status:
                        ax1.plot([0, x], [0, y], 'g--', alpha=0.3, linewidth=1)
                
                # ax1.set_xlim(-1.2, 1.2)
                # ax1.set_ylim(-1.2, 1.2)
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Y Position')
                ax1.set_title(f'Step {current_step["step"]} - Reward: {current_step["reward"]:.3f}')
                ax1.grid(True, alpha=0.3)
                
                # Plot metrics up to current step
                steps_so_far = steps[:frame+1]
                step_nums = [s['step'] for s in steps_so_far]
                rewards = [s['reward'] for s in steps_so_far]
                cumulative_rewards = [s['cumulative_reward'] for s in steps_so_far]
                served_counts = [sum(1 for u in s['users_data'] if u['action']['serve_binary']) for s in steps_so_far]
                
                ax2.plot(step_nums, rewards, 'b-', label='Step Reward', linewidth=2)
                ax2.plot(step_nums, cumulative_rewards, 'r-', label='Cumulative Reward', linewidth=2)
                ax2_twin = ax2.twinx()
                ax2_twin.plot(step_nums, served_counts, 'g-', label='Users Served', linewidth=2)
                ax2_twin.set_ylabel('Users Served')
                
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Reward')
                ax2.set_title(f'Metrics Progress - Episode {episode_data["episode"]}')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
                
            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=500, repeat=True)
            
            # Save as GIF
            gif_filename = f"user_movement{filename_suffix}.gif"
            gif_path = os.path.join(self.save_dir, gif_filename)
            
            # Use PillowWriter for better compatibility
            writer = animation.PillowWriter(fps=2)
            anim.save(gif_path, writer=writer)
            plt.close()
            
            print(f"🎬 User movement GIF saved: {gif_path}")
            return gif_path
            
        except Exception as e:
            print(f"Warning: Failed to create user movement GIF: {e}")
            return None
    
    def create_metrics_progression_gif(self, episode_data, filename_suffix=""):
        """Create GIF showing metrics progression throughout an episode"""
        try:
            steps = episode_data['steps']
            if not steps:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Metrics Progression - Episode {episode_data["episode"]}', fontsize=16)
            
            def animate(frame):
                for ax in axes.flat:
                    ax.clear()
                
                # Data up to current frame
                steps_so_far = steps[:frame+1]
                step_nums = [s['step'] for s in steps_so_far]
                
                # Extract metrics
                rewards = [s['reward'] for s in steps_so_far]
                cumulative_rewards = [s['cumulative_reward'] for s in steps_so_far]
                served_counts = [sum(1 for u in s['users_data'] if u['action']['serve_binary']) for s in steps_so_far]
                total_latencies = [s['system_totals']['total_latency'] for s in steps_so_far]
                total_flops = [s['system_totals']['total_flops'] for s in steps_so_far]
                total_mem = [s['system_totals']['total_mem'] for s in steps_so_far]
                
                # Plot 1: Rewards
                axes[0, 0].plot(step_nums, rewards, 'b-', linewidth=2, label='Step Reward')
                axes[0, 0].plot(step_nums, cumulative_rewards, 'r-', linewidth=2, label='Cumulative')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].set_title(f'Rewards (Step {frame+1}/{len(steps)})')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                
                # Plot 2: Users Served
                axes[0, 1].plot(step_nums, served_counts, 'g-', linewidth=2, marker='o')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Users Served')
                axes[0, 1].set_title('Users Served per Step')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: System Resources
                axes[1, 0].plot(step_nums, total_latencies, 'orange', linewidth=2, label='Latency')
                axes[1, 0].plot(step_nums, total_flops, 'purple', linewidth=2, label='FLOPS')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Resource Usage')
                axes[1, 0].set_title('System Resources')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                
                # Plot 4: Memory Usage
                axes[1, 1].plot(step_nums, total_mem, 'm-', linewidth=2, marker='s')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Memory Usage')
                axes[1, 1].set_title('Memory Consumption')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add current step indicator
                current_step = frame
                for ax in axes.flat:
                    ax.axvline(x=current_step, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=500, repeat=True)
            
            # Save as GIF
            gif_filename = f"metrics_progression{filename_suffix}.gif"
            gif_path = os.path.join(self.save_dir, gif_filename)
            
            writer = animation.PillowWriter(fps=2)
            anim.save(gif_path, writer=writer)
            plt.close()
            
            print(f"📊 Metrics progression GIF saved: {gif_path}")
            return gif_path
            
        except Exception as e:
            print(f"Warning: Failed to create metrics progression GIF: {e}")
            return None
    
    def create_combined_episode_gif(self, episode_data, filename_suffix=""):
        """Create a comprehensive GIF combining user movements and metrics"""
        try:
            steps = episode_data['steps']
            if not steps:
                return
                
            fig = plt.figure(figsize=(20, 12))
            
            # Create grid layout
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # User position plot (large, left side)
            ax_pos = fig.add_subplot(gs[:2, :2])
            
            # Metrics plots (right side)
            ax_reward = fig.add_subplot(gs[0, 2])
            ax_served = fig.add_subplot(gs[1, 2])
            ax_resources = fig.add_subplot(gs[2, :])
            
            def animate(frame):
                # Clear all axes
                ax_pos.clear()
                ax_reward.clear()
                ax_served.clear()
                ax_resources.clear()
                
                current_step = steps[frame]
                steps_so_far = steps[:frame+1]
                step_nums = [s['step'] for s in steps_so_far]
                
                # === USER POSITIONS ===
                # Service provider
                ax_pos.scatter(0, 0, c='blue', s=300, marker='s', label='Service Provider', 
                              edgecolors='black', linewidth=2, zorder=10)
                
                # Users with trails
                colors = plt.cm.Set1(np.linspace(0, 1, len(current_step['users_data'])))
                for i, user in enumerate(current_step['users_data']):
                    x, y = user['position']['x'], user['position']['y']
                    serve_status = user['action']['serve_binary']
                    denoise_steps = user['action']['denoise_steps']
                    
                    # Draw trail for this user across all previous steps
                    user_positions_x = [s['users_data'][i]['position']['x'] for s in steps_so_far]
                    user_positions_y = [s['users_data'][i]['position']['y'] for s in steps_so_far]
                    ax_pos.plot(user_positions_x, user_positions_y, color=colors[i], alpha=0.3, linewidth=1)
                    
                    # Current position
                    size = max(150, denoise_steps * 30) if serve_status else 80
                    marker = 'o' if serve_status else 'x'
                    ax_pos.scatter(x, y, c=colors[i], s=size, marker=marker, alpha=0.8,
                                  edgecolors='black', linewidth=1, zorder=5)
                    
                    # Connection line if served
                    if serve_status:
                        ax_pos.plot([0, x], [0, y], color=colors[i], linestyle='--', alpha=0.6, linewidth=2)
                
                ax_pos.set_xlim(-1.3, 1.3)
                ax_pos.set_ylim(-1.3, 1.3)
                ax_pos.set_xlabel('X Position')
                ax_pos.set_ylabel('Y Position')
                ax_pos.set_title(f'User Movements & Service Status\nStep {current_step["step"]} - Reward: {current_step["reward"]:.3f}')
                ax_pos.grid(True, alpha=0.3)
                
                # === REWARDS ===
                rewards = [s['reward'] for s in steps_so_far]
                cumulative_rewards = [s['cumulative_reward'] for s in steps_so_far]
                
                ax_reward.plot(step_nums, rewards, 'b-', linewidth=2, label='Step')
                ax_reward.plot(step_nums, cumulative_rewards, 'r-', linewidth=2, label='Cumulative')
                ax_reward.set_xlabel('Step')
                ax_reward.set_ylabel('Reward')
                ax_reward.set_title('Rewards')
                ax_reward.grid(True, alpha=0.3)
                ax_reward.legend()
                
                # === USERS SERVED ===
                served_counts = [sum(1 for u in s['users_data'] if u['action']['serve_binary']) for s in steps_so_far]
                avg_denoise_steps = []
                for s in steps_so_far:
                    served_users = [u for u in s['users_data'] if u['action']['serve_binary']]
                    avg_steps = np.mean([u['action']['denoise_steps'] for u in served_users]) if served_users else 0
                    avg_denoise_steps.append(avg_steps)
                
                ax_served.plot(step_nums, served_counts, 'g-', linewidth=2, marker='o', label='Count')
                ax_served_twin = ax_served.twinx()
                ax_served_twin.plot(step_nums, avg_denoise_steps, 'orange', linewidth=2, marker='s', label='Avg Steps')
                ax_served.set_xlabel('Step')
                ax_served.set_ylabel('Users Served')
                ax_served_twin.set_ylabel('Avg Denoise Steps')
                ax_served.set_title('Service Statistics')
                ax_served.grid(True, alpha=0.3)
                ax_served.legend(loc='upper left')
                ax_served_twin.legend(loc='upper right')
                
                # === SYSTEM RESOURCES ===
                total_latencies = [s['system_totals']['total_latency'] for s in steps_so_far]
                total_flops = [s['system_totals']['total_flops'] for s in steps_so_far]
                total_mem = [s['system_totals']['total_mem'] for s in steps_so_far]
                
                ax_resources.plot(step_nums, total_latencies, 'orange', linewidth=2, label='Latency', marker='o')
                ax_resources.plot(step_nums, total_flops, 'purple', linewidth=2, label='FLOPS', marker='s')
                ax_resources.plot(step_nums, total_mem, 'm', linewidth=2, label='Memory', marker='^')
                ax_resources.set_xlabel('Step')
                ax_resources.set_ylabel('Resource Usage')
                ax_resources.set_title('System Resources Usage')
                ax_resources.grid(True, alpha=0.3)
                ax_resources.legend()
                
                # Add step indicator
                for ax in [ax_reward, ax_served, ax_resources]:
                    ax.axvline(x=frame, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=len(steps), interval=800, repeat=True)
            
            # Save as GIF
            gif_filename = f"combined{filename_suffix}.gif"
            gif_path = os.path.join(self.save_dir, gif_filename)
            
            writer = animation.PillowWriter(fps=1.25)
            anim.save(gif_path, writer=writer)
            plt.close()
            
            print(f"🎥 Combined episode GIF saved: {gif_path}")
            return gif_path
            
        except Exception as e:
            print(f"Warning: Failed to create combined episode GIF: {e}")
            return None
    
    def save_episode_gifs(self, episode_indices=None):
        """Save GIFs for specified episodes or all recorded episodes"""
        if not self.gif_episodes_data:
            print("No episode data available for GIF creation")
            return
            
        episodes_to_process = self.gif_episodes_data
        if episode_indices:
            episodes_to_process = [self.gif_episodes_data[i] for i in episode_indices 
                                 if i < len(self.gif_episodes_data)]
        
        print(f"🎬 Creating GIFs for {len(episodes_to_process)} episodes...")
        
        created_gifs = []
        for episode_data in episodes_to_process:
            # Create all three types of GIFs
            gif1 = self.create_user_movement_gif(episode_data)
            gif2 = self.create_metrics_progression_gif(episode_data)
            gif3 = self.create_combined_episode_gif(episode_data)
            
            created_gifs.extend([gif for gif in [gif1, gif2, gif3] if gif])
        
        print(f"✅ Created {len(created_gifs)} GIF files")
        return created_gifs

    def save_user_positions_gif(self, episode_index=-1):
        """Create GIF showing user positions step-by-step for a given episode"""
        if len(self.env_metrics) == 0:
            return
    
        try:
            # Lấy dữ liệu của episode (mặc định: cuối cùng)
            episode_data = self.gif_episodes_data[episode_index] if self.gif_episodes_data else None
            if not episode_data:
                print("⚠️ No step-by-step data available for GIF")
                return
    
            steps = episode_data["steps"]
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
            def animate(frame):
                ax.clear()
                step_data = steps[frame]
                for user in step_data['users_data']:
                    x, y = user['position']['x'], user['position']['y']
                    serve_status = user['action']['serve_binary']
                    denoise_steps = user['action']['denoise_steps']
    
                    color = 'green' if serve_status else 'red'
                    size = max(50, denoise_steps * 10) if serve_status else 30
                    ax.scatter(x, y, c=color, s=size, alpha=0.7)
    
                # Service provider at origin
                ax.scatter(0, 0, c='blue', s=200, marker='s',
                        label='Service Provider', edgecolors='black', linewidth=2)
    
                # ax.set_xlim(-1.2, 1.2)
                # ax.set_ylim(-1.2, 1.2)
                ax.set_title(f"Episode {episode_data['episode']} - Step {frame}")
                ax.grid(True, alpha=0.3)
    
            anim = animation.FuncAnimation(fig, animate, frames=len(steps),
                                        interval=500, repeat=True)
    
            gif_path = os.path.join(self.save_dir, f"user_positions.gif")
            writer = animation.PillowWriter(fps=2)
            anim.save(gif_path, writer=writer)
            plt.close()
    
            print(f"📍 User positions GIF saved: {gif_path}")
            return gif_path

        except Exception as e:
            print(f"Warning: Failed to create user positions GIF: {e}")


    def log_env_metrics(self, episode, env_info, action, state, config):
        """Log environment-specific metrics including user data, diffusion steps, etc."""
        try:
            num_users = config["num_users"]
            
            # Extract system totals from env_info
            total_served = env_info.get('total_served', 0)
            total_latency = env_info.get('total_latency', 0)
            total_flops = env_info.get('total_flops', 0)
            total_mem = env_info.get('total_mem', 0)
            penalty_qos = env_info.get('penalty_qos', 0)
            qos = env_info.get('mean_qos', 0)
            latency = env_info.get('latency', 0)
            flops = env_info.get('flops', 0)
            mem = env_info.get('mem', 0)
            price = env_info.get('price', 0)
            denoise_steps = env_info.get('denoise_steps', [0]*num_users)
            bonus = env_info.get('bonus', 0)
            penalty = env_info.get('penalty', 0)
            
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
                    'total_served': int(total_served),
                    'total_latency': float(total_latency),
                    'total_flops': float(total_flops),
                    'total_mem': float(total_mem),
                    'bonus': float(bonus),
                    'penalty': float(penalty),
                    'penalty_qos': float(penalty_qos),
                    'mean_qos': float(qos),
                    'latency': float(latency),
                    'flops': float(flops),
                    'mem': float(mem),
                    'price': float(price),
                    'denoise_steps': [int(ds) for ds in denoise_steps] if isinstance(denoise_steps, (list, np.ndarray)) else [0]*num_users
                },
                'constraints_status': {
                    'latency_ok': bool(total_latency <= config["sys_tau"] * num_users),
                    'flops_ok': bool(total_flops <= config["Gmax"]),
                    'mem_ok': bool(total_mem <= config["Mmax"])
                }
            }
            
            # Debug: Print system totals to verify they're being captured
            if episode % 10 == 0:  # Print every 10th episode to avoid spam
                print(f"DEBUG Episode {episode}: served={total_served}, latency={total_latency:.2f}, flops={total_flops:.2f}, mem={total_mem:.2f}")
            
            # Extract user-specific information from state and action
            for i in range(num_users):
                # State format: [x, y, image_size, prompt_size, direction, qos_required] per user
                user_state_start = i * 6
                
                # Get raw actions - handle potential array format
                try:
                    raw_serve_decision = float(action[2*i])
                    raw_denoise_action = float(action[2*i + 1])
                except (IndexError, TypeError) as e:
                    print(f"Warning: Action format issue for user {i}: {e}")
                    raw_serve_decision = 0.0
                    raw_denoise_action = 0.0
                
                # Normalize actions if they are outside [0,1] range
                # Assuming action space is [-1, 1], normalize to [0, 1]
                normalized_serve = (raw_serve_decision + 1.0) / 2.0 if raw_serve_decision < 0 or raw_serve_decision > 1 else raw_serve_decision
                normalized_denoise = (raw_denoise_action + 1.0) / 2.0 if raw_denoise_action < 0 or raw_denoise_action > 1 else raw_denoise_action
                
                # Clip to [0, 1] to be safe
                normalized_serve = np.clip(normalized_serve, 0.0, 1.0)
                normalized_denoise = np.clip(normalized_denoise, 0.0, 1.0)
                
                # Scale denoise action from [0,1] to [1, max_denoise_steps]
                scaled_denoise_steps = int(1 + normalized_denoise * (config["max_denoise_steps"] - 1))
                scaled_denoise_steps = np.clip(scaled_denoise_steps, 1, config["max_denoise_steps"])
                
                # Get user state with bounds checking
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
                            'raw_serve_action': float(raw_serve_decision),
                            'raw_denoise_action': float(raw_denoise_action),
                            'normalized_denoise': float(normalized_denoise)
                        }
                    }
                    env_data['users_data'].append(user_data)
                except (IndexError, TypeError) as e:
                    print(f"Warning: State format issue for user {i}: {e}")
                    continue
            
            self.env_metrics.append(env_data)
            
        except Exception as e:
            print(f"Warning: Failed to log environment metrics: {e}")
            print(f"Debug - env_info keys: {list(env_info.keys()) if isinstance(env_info, dict) else 'not dict'}")
            print(f"Debug - action type: {type(action)}, shape: {getattr(action, 'shape', 'no shape')}")
            print(f"Debug - state type: {type(state)}, shape: {getattr(state, 'shape', 'no shape')}")
            
            # Create minimal log entry even if there are errors
            try:
                env_data = {
                    'episode': episode,
                    'timestamp': datetime.now().isoformat(),
                    'system_constraints': {
                        'Gmax': float(config.get("Gmax", 0)),
                        'Mmax': float(config.get("Mmax", 0)),
                        'sys_tau': float(config.get("sys_tau", 0)),
                        'max_denoise_steps': int(config.get("max_denoise_steps", 1))
                    },
                    'users_data': [],
                    'system_totals': {
                        'total_served': int(env_info.get('served', 0) if isinstance(env_info, dict) else 0),
                        'total_latency': float(env_info.get('total_latency', 0) if isinstance(env_info, dict) else 0),
                        'total_flops': float(env_info.get('total_flops', 0) if isinstance(env_info, dict) else 0),
                        'total_mem': float(env_info.get('total_mem', 0) if isinstance(env_info, dict) else 0),
                        'bonus': float(env_info.get('bonus', 0) if isinstance(env_info, dict) else 0),
                        'penalty': float(env_info.get('penalty', 0) if isinstance(env_info, dict) else 0),
                        'penalty_qos': float(env_info.get('penalty_qos', 0) if isinstance(env_info, dict) else 0),
                        'mean_qos': float(env_info.get('mean_qos', 0) if isinstance(env_info, dict) else 0),
                        'latency': float(env_info.get('latency', 0) if isinstance(env_info, dict) else 0),
                        'flops': float(env_info.get('flops', 0) if isinstance(env_info, dict) else 0),
                        'mem': float(env_info.get('mem', 0) if isinstance(env_info, dict) else 0),
                        'price': float(env_info.get('price', 0) if isinstance(env_info, dict) else 0),
                        'denoise_steps': [0]*config.get("num_users", 1) if isinstance(config.get("num_users", 1), int) else [0]
                    },
                    'constraints_status': {
                        'latency_ok': False,
                        'flops_ok': False,
                        'mem_ok': False
                    },
                    'error': str(e)
                }
                self.env_metrics.append(env_data)
            except Exception as e2:
                print(f"Failed to create minimal log entry: {e2}")

# ...existing code...
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
        final_plot_filename = f"final_training_plot.png"
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
        """Save all metrics to a single comprehensive CSV file"""
        if not self.epoch_metrics and not self.env_metrics:
            print("No metrics data to save")
            return
        
        # Create comprehensive metrics by merging epoch and environment data
        comprehensive_metrics = []
        
        # Create a mapping of episode numbers to environment data for quick lookup
        env_data_by_episode = {}
        if self.env_metrics:
            for env_data in self.env_metrics:
                episode = env_data['episode']
                if episode not in env_data_by_episode:
                    env_data_by_episode[episode] = {
                        'system_constraints': env_data['system_constraints'],
                        'system_totals': env_data['system_totals'],
                        'constraints_status': env_data['constraints_status'],
                        'users_data': []
                    }
                env_data_by_episode[episode]['users_data'].extend(env_data['users_data'])
        
        # Process each epoch metric and merge with corresponding environment data
        for epoch_data in self.epoch_metrics:
            episode = epoch_data['epoch']
            
            # Start with epoch data
            merged_row = epoch_data.copy()
            
            # Add environment data if available for this episode
            if episode in env_data_by_episode:
                env_episode_data = env_data_by_episode[episode]
                
                # Add system constraints
                merged_row.update({
                    'Gmax': env_episode_data['system_constraints']['Gmax'],
                    'Mmax': env_episode_data['system_constraints']['Mmax'],
                    'sys_tau': env_episode_data['system_constraints']['sys_tau'],
                    'max_denoise_steps': env_episode_data['system_constraints']['max_denoise_steps']
                })
                
                # Add system totals
                merged_row.update({
                    'total_served': env_episode_data['system_totals']['total_served'],
                    'total_latency': env_episode_data['system_totals']['total_latency'],
                    'total_flops': env_episode_data['system_totals']['total_flops'],
                    'total_mem': env_episode_data['system_totals']['total_mem'],
                    'bonus': env_episode_data['system_totals']['bonus'],
                    'penalty': env_episode_data['system_totals']['penalty'],
                    'penalty_qos': env_episode_data['system_totals']['penalty_qos'],
                    'mean_qos': env_episode_data['system_totals']['mean_qos'],
                    'latency': env_episode_data['system_totals']['latency'],
                    'flops': env_episode_data['system_totals']['flops'],
                    'mem': env_episode_data['system_totals']['mem'],
                    'price': env_episode_data['system_totals']['price'],
                    'denoise_steps': env_episode_data['system_totals']['denoise_steps']
                })
                
                # Add constraints status
                merged_row.update({
                    'latency_ok': env_episode_data['constraints_status']['latency_ok'],
                    'flops_ok': env_episode_data['constraints_status']['flops_ok'],
                    'mem_ok': env_episode_data['constraints_status']['mem_ok']
                })
                
                # Calculate aggregated user metrics
                users_data = env_episode_data['users_data']
                if users_data:
                    served_users = [u for u in users_data if u['action']['serve_binary'] == 1]
                    
                    merged_row.update({
                        'num_users': len(users_data),
                        'num_served': len(served_users),
                        'serve_ratio': len(served_users) / len(users_data) if users_data else 0,
                        'avg_user_distance': np.mean([np.sqrt(u['position']['x']**2 + u['position']['y']**2) for u in users_data]),
                        'avg_image_size': np.mean([u['image_size'] for u in users_data]),
                        'avg_prompt_size': np.mean([u['prompt_size'] for u in users_data]),
                        'avg_qos_required': np.mean([u['qos_required'] for u in users_data]),
                        'avg_denoise_steps_served': np.mean([u['action']['denoise_steps'] for u in served_users]) if served_users else 0,
                        'min_denoise_steps': min([u['action']['denoise_steps'] for u in served_users]) if served_users else 0,
                        'max_denoise_steps': max([u['action']['denoise_steps'] for u in served_users]) if served_users else 0
                    })
            else:
                # Fill with default values if no environment data
                merged_row.update({
                    'Gmax': None, 'Mmax': None, 'sys_tau': None, 'max_denoise_steps': None,
                    'total_served': None, 'total_latency': None, 'total_flops': None, 'total_mem': None,
                    'bonus': None, 'penalty': None, 'penalty_qos': None, 'mean_qos': None, 'latency': None,
                    'flops': None, 'mem': None, 'price': None, 'denoise_steps': None,
                    'latency_ok': None, 'flops_ok': None, 'mem_ok': None,
                    'num_users': None, 'num_served': None, 'serve_ratio': None,
                    'avg_user_distance': None, 'avg_image_size': None, 'avg_prompt_size': None,
                    'avg_qos_required': None, 'avg_denoise_steps_served': None,
                    'min_denoise_steps': None, 'max_denoise_steps': None
                })
        
            comprehensive_metrics.append(merged_row)
        
        # Save comprehensive metrics to CSV
        if comprehensive_metrics:
            comprehensive_file = os.path.join(self.save_dir, "comprehensive_metrics.csv")
            with open(comprehensive_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=comprehensive_metrics[0].keys())
                writer.writeheader()
                writer.writerows(comprehensive_metrics)
            print(f"Comprehensive metrics saved to: {comprehensive_file}")
        
        # Also save detailed user data separately if needed
        if self.env_metrics:
            detailed_user_metrics = []
            for env_data in self.env_metrics:
                base_info = {
                    'episode': env_data['episode'],
                    'timestamp': env_data['timestamp'],
                    'total_served': env_data['system_totals']['total_served'],
                    'total_latency': env_data['system_totals']['total_latency'],
                    'total_flops': env_data['system_totals']['total_flops'],
                    'total_mem': env_data['system_totals']['total_mem'],
                    'bonus': env_data['system_totals']['bonus'],
                    'penalty': env_data['system_totals']['penalty'],
                    'Gmax': env_data['system_constraints']['Gmax'],
                    'Mmax': env_data['system_constraints']['Mmax'],
                    'sys_tau': env_data['system_constraints']['sys_tau'],
                    'max_denoise_steps': env_data['system_constraints']['max_denoise_steps']
                }
                
                for user in env_data['users_data']:
                    user_row = base_info.copy()
                    user_row.update({
                        'user_id': user['user_id'],
                        'pos_x': user['position']['x'],
                        'pos_y': user['position']['y'],
                        'distance_from_sp': np.sqrt(user['position']['x']**2 + user['position']['y']**2),
                        'image_size': user['image_size'],
                        'prompt_size': user['prompt_size'],
                        'direction': user['direction'],
                        'qos_required': user['qos_required'],
                        'serve_decision': user['action']['serve_decision'],
                        'serve_binary': user['action']['serve_binary'],
                        'denoise_steps': user['action']['denoise_steps']
                    })
                    detailed_user_metrics.append(user_row)
            
            if detailed_user_metrics:
                detailed_file = os.path.join(self.save_dir, "detailed_user_metrics.csv")
                with open(detailed_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=detailed_user_metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(detailed_user_metrics)
                print(f"Detailed user metrics saved to: {detailed_file}")

    def save_to_json(self):
        """Save all metrics to JSON files"""
        # Save epoch metrics
        if self.epoch_metrics:
            epoch_file = os.path.join(self.save_dir, f"epoch_metrics.json")
            with open(epoch_file, 'w') as f:
                json.dump(self._convert_for_json(self.epoch_metrics), f, indent=2)
            print(f"Epoch metrics (JSON) saved to: {epoch_file}")
        
        # Save environment metrics
        if self.env_metrics:
            env_file = os.path.join(self.save_dir, f"environment_metrics.json")
            with open(env_file, 'w') as f:
                json.dump(self._convert_for_json(self.env_metrics), f, indent=2)
            print(f"Environment metrics (JSON) saved to: {env_file}")
    
    def save_env_metrics_plot(self):
        """Create plots for environment-specific metrics"""
        if len(self.env_metrics) == 0:
            print("No environment metrics to plot")
            return
            
        try:
            # Extract data for plotting with fallbacks
            episodes = []
            served_users = []
            total_latency = []
            total_flops = []
            total_mem = []
            bonuses = []
            penalties = []
            avg_denoise_steps = []
            qos_compliance = []
            mean_qos = []
            
            for m in self.env_metrics:
                episodes.append(m['episode'])
                users_data = m.get('system_totals', [])
                if users_data:
                    mean_qos.append(users_data['mean_qos'])
                    compliant_count = users_data['mean_qos']
                    qos_compliance.append(compliant_count / len(users_data))
                else:
                    mean_qos.append(0) 
                    qos_compliance.append(0)  
                # Handle both new format (with system_totals) and old format (without)
                if 'system_totals' in m:
                    served_users.append(m['system_totals'].get('total_served', 0))
                    total_latency.append(m['system_totals'].get('total_latency', 0))
                    total_flops.append(m['system_totals'].get('total_flops', 0))
                    total_mem.append(m['system_totals'].get('total_mem', 0))
                    bonuses.append(m['system_totals'].get('bonus', 0))
                    penalties.append(m['system_totals'].get('penalty', 0))
                else:
                    # Fallback: calculate from user data
                    users_data = m.get('users_data', [])
                    served_count = sum(1 for u in users_data if u['action']['serve_binary'] == 1)
                    served_users.append(served_count)
                    total_latency.append(0)  # Cannot calculate without system info
                    total_flops.append(0)    # Cannot calculate without system info
                    total_mem.append(0)      # Cannot calculate without system info  
                    bonuses.append(0)        # Cannot calculate without system info
                    penalties.append(0)      # Cannot calculate without system info
                
                # Calculate average denoise steps
                users_data = m.get('users_data', [])
                steps = [user['action']['denoise_steps'] for user in users_data if user['action']['serve_binary'] == 1]
                avg_steps = np.mean(steps) if steps else 0
                avg_denoise_steps.append(avg_steps)
            
            # Create figure with subplots
            fig, axes = plt.subplots(4, 2, figsize=(15, 12))
            fig.suptitle('Environment Metrics Over Episodes', fontsize=16)
            
            # Plot 1: Served Users
            axes[0, 0].plot(episodes, served_users, 'g-', linewidth=2, label='Served Users')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Number of Users Served')
            axes[0, 0].set_title('Users Served per Episode')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: System Resources (only if we have data)
            if any(total_latency):
                axes[0, 1].plot(episodes, total_latency, 'r-', linewidth=2, label='Total Latency')
                axes[0, 1].plot(episodes, [config["sys_tau"] for m in self.env_metrics], 'k--', linewidth=1, label='Latency Constraint')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Total Latency')
                axes[0, 1].set_title('System Latency')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'Latency data not available\n(system_totals missing)', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('System Latency - No Data')
            
            # Plot 3: Computational Resources (only if we have data)
            if any(total_flops):
                axes[1, 0].plot(episodes, total_flops, 'b-', linewidth=2, label='Total FLOPS')
                axes[1, 0].axhline(y=config["Gmax"], color='k', linestyle='--', linewidth=1, label='FLOPS Constraint')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Total FLOPS')
                axes[1, 0].set_title('Computational Load (FLOPS)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'FLOPS data not available\n(system_totals missing)', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Computational Load (FLOPS) - No Data')
            
            # Plot 4: Memory Usage (only if we have data)
            if any(total_mem):
                axes[1, 1].plot(episodes, total_mem, 'm-', linewidth=2, label='Total Memory')
                axes[1, 1].axhline(y=config["Mmax"], color='k', linestyle='--', linewidth=1, label='Memory Constraint')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Total Memory')
                axes[1, 1].set_title('Memory Usage')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            else:
                axes[1, 1].text(0.5, 0.5, 'Memory data not available\n(system_totals missing)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Memory Usage - No Data')
            
            # Plot 5: Rewards and Penalties (only if we have data)
            if any(bonuses) or any(penalties):
                axes[2, 0].plot(episodes, bonuses, 'g-', linewidth=2, label='Bonus')
                axes[2, 0].plot(episodes, penalties, 'r-', linewidth=2, label='Penalty')
                axes[2, 0].set_xlabel('Episode')
                axes[2, 0].set_ylabel('Value')
                axes[2, 0].set_title('Bonus vs Penalty')
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].legend()
            else:
                axes[2, 0].text(0.5, 0.5, 'Bonus/Penalty data not available\n(system_totals missing)', 
                               ha='center', va='center', transform=axes[2, 0].transAxes)
                axes[2, 0].set_title('Bonus vs Penalty - No Data')
            
            # Plot 6: Average Diffusion Steps (this should always be available)
            if avg_denoise_steps:
                axes[2, 1].plot(episodes, avg_denoise_steps, 'orange', linewidth=2, label='Avg Denoise Steps')
                # print("DEBUG: avg_denoise_steps:", avg_denoise_steps)
                axes[2, 1].set_xlabel('Episode')
                axes[2, 1].set_ylabel('Average Steps')
                axes[2, 1].set_title('Average Diffusion Steps (Served Users)')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].legend()
                
                # Add statistics
                if avg_denoise_steps:
                    stats_text = f'Mean: {np.mean(avg_denoise_steps):.1f}\n'
                    stats_text += f'Max: {np.max(avg_denoise_steps):.1f}\n'
                    stats_text += f'Min: {np.min([x for x in avg_denoise_steps if x > 0]):.1f}'
                    axes[2, 1].text(0.02, 0.98, stats_text, transform=axes[2, 1].transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[2, 1].text(0.5, 0.5, 'No diffusion steps data available', 
                               ha='center', va='center', transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Average Diffusion Steps - No Data')
            

            # Plot 7: QoS Compliance Rate (if qos data available)  
            if any(qos_compliance):
                axes[3, 0].plot(episodes, qos_compliance, 'purple', linewidth=2, label='QoS Compliance Rate')
                axes[3, 0].set_xlabel('Episode')
                axes[3, 0].set_ylabel('Compliance Rate')
                axes[3, 0].set_title('QoS Compliance Rate')
                axes[3, 0].grid(True, alpha=0.3)
                axes[3, 0].legend()
            else:
                axes[3, 0].text(0.5, 0.5, 'QoS data not available', 
                              ha='center', va='center', transform=axes[3, 0].transAxes)
                axes[3, 0].set_title('QoS Compliance Rate - No Data')


                       
            if any(mean_qos):
                axes[3, 1].plot(episodes, mean_qos, 'purple', linewidth=2, label='Average QoS')
                axes[3, 1].axhline(y=np.mean(config.get("qos_required", 0)), color='k', linestyle='--', linewidth=1, label='Avg QoS Required')
                axes[3, 1].set_xlabel('Episode')
                axes[3, 1].set_ylabel('Average QoS')
                axes[3, 1].set_title("Average QoS")
                axes[3, 1].grid(True, alpha=0.3)
                axes[3, 1].legend()
            else:
                axes[3, 1].text(0.5, 0.5, 'QoS data not available', 
                              ha='center', va='center', transform=axes[3, 0].transAxes)
                axes[3, 1].set_title('Average QoS - No Data')


            plt.tight_layout()
            # Save plot
            env_plot_path = os.path.join(self.save_dir, "environment_metrics.png")
            plt.savefig(env_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Environment metrics plot saved: {env_plot_path}")
            
            # Print diagnostic info
            has_system_totals = any('system_totals' in m for m in self.env_metrics)
            print(f"DEBUG: Environment metrics contains system_totals: {has_system_totals}")
            print(f"DEBUG: Total episodes logged: {len(self.env_metrics)}")
            if self.env_metrics:
                sample_keys = list(self.env_metrics[0].keys())
                print(f"DEBUG: Sample metric keys: {sample_keys}")
            
        except Exception as e:
            print(f"Warning: Failed to create environment metrics plot: {e}")
            import traceback
            traceback.print_exc()
    
    def save_user_positions_plot(self):
        """Create a plot showing user positions over time"""
        if len(self.env_metrics) == 0:
            return
            
        try:
            # Create user trajectory plot for the last episode
            last_episode_data = self.env_metrics[-1]
            users_data = last_episode_data['users_data']
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot user positions
            for user in users_data:
                x, y = user['position']['x'], user['position']['y']
                serve_status = user['action']['serve_binary']
                denoise_steps = user['action']['denoise_steps']
                
                color = 'green' if serve_status else 'red'
                size = max(50, denoise_steps * 10) if serve_status else 30
                
                ax.scatter(x, y, c=color, s=size, alpha=0.7, 
                          label=f"User {user['user_id']}: {'Served' if serve_status else 'Not Served'}")
            
            # Add service provider position
            sp_pos = [0.0, 0.0]  # Assuming SP is at origin, adjust if needed
            ax.scatter(sp_pos[0], sp_pos[1], c='blue', s=200, marker='s', 
                      label='Service Provider', edgecolors='black', linewidth=2)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'User Positions and Service Status (Episode {last_episode_data["episode"]})')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            pos_plot_path = os.path.join(self.save_dir, "user_positions.png")
            plt.savefig(pos_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📍 User positions plot saved: {pos_plot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create user positions plot: {e}")

hyperparameters = {
    'gail-service-env':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v1':      {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v3-org':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 25, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v4':      {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v5':      {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 100, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1}
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


    evaluations = []
    utils.print_banner(f"Training Start", separator="*", num_star=90)

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    
    metrics_logger = MetricsLogger(save_dir=os.path.join(output_dir, "metrics"))

    episode_rewards = []  
    current_state = env.reset()  # Initialize state

    for episode in tqdm(range(args.num_episodes)):
        state = env.reset()
        current_state = state  # Keep track of current state
        done = False
        episode_reward = 0  
        episode_losses = {}  
        saved_info = None
        saved_action = None  # Track the last action

        should_record_gif = (episode % args.num_episodes == 0)
        if should_record_gif:
            metrics_logger.start_episode_recording()
        
        step_count = 0
        while not done:
            action = agent.sample_action(state)
            saved_action = action  # Save action for metrics
            next_state, reward, done, info = env.step(action)
            
            # Print memory information if enabled
            if args.print_logger:
                total_mem = info.get('total_mem', 0)
                mem_max = config.get('Mmax', 0)
                print(f"Episode {episode}, Step {step_count}: Total Memory = {total_mem:.2f}, Memory Max = {mem_max:.2f}, Usage = {(total_mem/mem_max*100) if mem_max > 0 else 0:.1f}%")
                
                # Also print other resource info for debugging
                total_flops = info.get('total_flops', 0)
                flops_max = config.get('Gmax', 0)
                total_latency = info.get('total_latency', 0)
                latency_max = config.get('sys_tau', 0) * config.get('num_users', 1)
                served = info.get('served', 0)
                
                print(f"  FLOPS: {total_flops:.2f}/{flops_max:.2f} ({(total_flops/flops_max*100) if flops_max > 0 else 0:.1f}%)")
                print(f"  Latency: {total_latency:.2f}/{latency_max:.2f} ({(total_latency/latency_max*100) if latency_max > 0 else 0:.1f}%)")
                print(f"  Users Served: {served}/{config.get('num_users', 0)}")
                print(f"  Reward: {reward:.4f}")
                print("-" * 50)
            
            # Log step data for GIF
            if should_record_gif:
                metrics_logger.log_step_data(step_count, state, action, reward, info, config)
            
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            current_state = state  # Update current state
            episode_reward += reward  
            saved_info = info  
            step_count += 1

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
                    
        # End episode recording
        if should_record_gif:
            metrics_logger.end_episode_recording(episode, should_create_gif=True)
        
        # Print episode summary if enabled
        if args.print_logger and saved_info is not None:
            total_mem = saved_info.get('total_mem', 0)
            mem_max = config.get('Mmax', 0)
            total_flops = saved_info.get('total_flops', 0)
            flops_max = config.get('Gmax', 0)
            total_latency = saved_info.get('total_latency', 0)
            latency_max = config.get('sys_tau', 0) * config.get('num_users', 1)
            served = saved_info.get('served', 0)
            bonus = saved_info.get('bonus', 0)
            penalty = saved_info.get('penalty', 0)
            
            print(f"\n{'='*60}")
            print(f"EPISODE {episode} SUMMARY:")
            print(f"{'='*60}")
            print(f"Episode Reward: {episode_reward:.4f}")
            print(f"Episode Steps: {step_count}")
            print(f"Users Served: {served}/{config.get('num_users', 0)} ({(served/config.get('num_users', 1)*100):.1f}%)")
            
            # NEW: Dynamic scheduling info
            served_users = saved_info.get('served_users', [])
            if served_users:
                print(f"Served User IDs: {served_users}")
            print(f"Avg Denoise Steps: {saved_info.get('avg_denoise_steps', 0):.1f}")
            print(f"")
            print(f"RESOURCE UTILIZATION:")
            print(f"  Memory:  {total_mem:.2f}/{mem_max:.2f} ({saved_info.get('memory_utilization', 0)*100:.1f}%) {'✓' if total_mem <= mem_max else '✗'}")
            print(f"  FLOPS:   {total_flops:.2f}/{flops_max:.2f} ({saved_info.get('flops_utilization', 0)*100:.1f}%) {'✓' if total_flops <= flops_max else '✗'}")
            print(f"  Latency: {total_latency:.2f}/{latency_max:.2f} ({saved_info.get('latency_utilization', 0)*100:.1f}%) {'✓' if total_latency <= latency_max else '✗'}")
            print(f"")
            print(f"REWARD BREAKDOWN:")
            print(f"  Bonus:   {bonus:.4f}")
            print(f"  Penalty: {penalty:.4f}")
            print(f"  Net:     {bonus + penalty:.4f}")
            print(f"")
            print(f"DEBUG - saved_info keys: {list(saved_info.keys())}")
            print(f"DEBUG - config keys: {list(config.keys())}")
            print(f"{'='*60}\n")
                    
        episode_rewards.append(episode_reward)  
        
        # Log environment metrics
        if saved_info is not None and saved_action is not None:
            metrics_logger.log_env_metrics(episode, saved_info, saved_action, current_state, config)
        
        agent_episode_losses = None
        if episode_losses:
            agent_episode_losses = {}
            for loss_name, loss_values in episode_losses.items():
                if loss_values:
                    agent_episode_losses[f"avg_{loss_name}"] = np.mean(loss_values)
                    agent_episode_losses[f"std_{loss_name}"] = np.std(loss_values)
        
        reward_improved = metrics_logger.log_epoch_metrics(episode, episode_reward, agent_episode_losses)
        

            

        np.save(os.path.join(output_dir, f"episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
        
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes, config=config, eval_env=env)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            # np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']), # These are not available in online RL
                            # np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']), # These are not available in online RL
                            episode, # Use episode number for logging
                            ])
        if episode % args.eval_freq == 0:
            if reward_improved:
                print(f"🎯 New best reward achieved: {episode_reward:.4f} at episode {episode + 1}")
            metrics_logger.save_final_plot()
            metrics_logger.save_env_metrics_plot()
            metrics_logger.save_user_positions_plot()
            metrics_logger.save_user_positions_gif()
            # metrics_logger.save_to_csv()
            metrics_logger.save_to_json()
            
        # np.save(os.path.join(output_dir, "eval_res"), eval_res)
        # np.save(os.path.join(output_dir, "eval"), evaluations)
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
    metrics_logger.save_env_metrics_plot()  # Final environment metrics plot
    metrics_logger.save_user_positions_plot()  # Final user positions plot
    # metrics_logger.save_to_csv()
    metrics_logger.save_to_json()
    
    print(f"📊 Final training plots and metrics saved to: {metrics_logger.save_dir}")


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
    parser.add_argument("--algo", default="dql", type=str)  # ['bc', 'ql']
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
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'/{args.n_users}/{args.seed}'

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
        from env.env import GAIServiceEnv_v1 as GAIAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v5':
        from env.env_v5_org import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v4':
        from env.env_v4 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v3-org':
        from env.env_v3_org import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig

    config =  EnvConfig(args.env_name)
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

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args,
                config)
