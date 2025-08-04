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
import json
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
        
        # Chỉ save plot theo interval hoặc epoch cuối
        if current_epoch % save_interval != 0 and current_epoch != self.loss_history['epochs'][-1]:
            return
            
        # Create figure with subplots
        num_plots = 1 + len(self.loss_history['losses'])  # 1 for rewards + losses
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
        
        if num_plots == 1:
            axes = [axes]
            
        # Plot rewards
        axes[0].plot(self.loss_history['epochs'], self.loss_history['rewards'], 'b-', linewidth=2, label='Average Reward')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Progress - Average Reward per Epoch')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot losses
        plot_idx = 1
        for loss_name, loss_values in self.loss_history['losses'].items():
            if len(loss_values) > 0:
                # Ensure same length as epochs
                epochs_for_loss = self.loss_history['epochs'][-len(loss_values):]
                axes[plot_idx].plot(epochs_for_loss, loss_values, 'r-', linewidth=2, label=loss_name)
                axes[plot_idx].set_xlabel('Epoch')
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
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
        
        if num_plots == 1:
            axes = [axes]
            
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
            
        axes[0].set_xlabel('Epoch')
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
                
                axes[plot_idx].set_xlabel('Epoch')
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
        
    def log_step_metrics(self, epoch, episode, step, state, action, reward, info, agent_losses=None):
        """Log chi tiết metrics mỗi step"""
        step_data = {
            'epoch': epoch,
            'episode': episode,
            'step': step,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log action details
        if isinstance(action, dict):
            step_data['uav_movement'] = action.get('uav_movement', []).tolist() if hasattr(action.get('uav_movement', []), 'tolist') else action.get('uav_movement', [])
            step_data['denoising_steps'] = action.get('denoising_steps', []).tolist() if hasattr(action.get('denoising_steps', []), 'tolist') else action.get('denoising_steps', [])
            step_data['uav_movement_magnitude'] = np.linalg.norm(action.get('uav_movement', [0])) if action.get('uav_movement') is not None else 0
        else:
            step_data['action'] = action.tolist() if hasattr(action, 'tolist') else action
            
        # Log state summary
        if state is not None:
            step_data['state_mean'] = np.mean(state)
            step_data['state_std'] = np.std(state)
            step_data['state_min'] = np.min(state)
            step_data['state_max'] = np.max(state)
            
        # Log environment info metrics
        if info:
            # Energy metrics
            step_data['energy_total'] = info.get('energy_total_consumption', 0)
            step_data['energy_transmission'] = info.get('energy_transmission', 0)
            step_data['energy_processing'] = info.get('energy_processing', 0)
            step_data['energy_efficiency_per_ue'] = info.get('energy_efficiency_per_ue', 0)
            
            # Latency metrics
            step_data['latency_total_avg'] = info.get('latency_total_avg', 0)
            step_data['latency_upload_avg'] = info.get('latency_upload_avg', 0)
            step_data['latency_processing_avg'] = info.get('latency_processing_avg', 0)
            step_data['latency_download_avg'] = info.get('latency_download_avg', 0)
            step_data['latency_deadline_violations'] = info.get('latency_deadline_violations', 0)
            
            # QoS metrics
            step_data['qos_quality_score_avg'] = info.get('qos_quality_score_avg', 0)
            step_data['qos_denoising_efficiency'] = info.get('qos_denoising_efficiency', 0)
            step_data['qos_quality_requirements_met'] = info.get('qos_quality_requirements_met', 0)
            step_data['qos_quality_violations'] = info.get('qos_quality_violations', 0)
            
            # Completion metrics
            step_data['completion_rate'] = info.get('completion_rate', 0)
            step_data['genai_completion_rate'] = info.get('genai_completion_rate', 0)
            step_data['dnn_completion_rate'] = info.get('dnn_completion_rate', 0)
            
            # UE status
            step_data['total_genai_ues'] = info.get('total_genai_ues', 0)
            step_data['total_dnn_ues'] = info.get('total_dnn_ues', 0)
            step_data['current_processing_ue'] = info.get('current_processing_ue', -1)
            step_data['current_processing_type'] = info.get('current_processing_type', 'unknown')
            
        # Add agent losses if available
        if agent_losses:
            step_data.update(agent_losses)
            
        self.step_metrics.append(step_data)
        
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
            
        # Save step metrics  
        if self.step_metrics:
            step_file = os.path.join(self.save_dir, f"step_metrics.csv")
            with open(step_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.step_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.step_metrics)
            print(f"Step metrics saved to: {step_file}")
    
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
        """Save all metrics to JSON files"""
        # print(self.epoch_metrics)
        # print(self.step_metrics[0])
        # Save epoch metrics
        if self.epoch_metrics:
            epoch_file = os.path.join(self.save_dir, f"epoch_metrics.json")
            with open(epoch_file, 'w') as f:
                json.dump(self._convert_for_json(self.epoch_metrics), f, indent=2)
            print(f"Epoch metrics (JSON) saved to: {epoch_file}")
            
        # Save step metrics
        if self.step_metrics:
            step_file = os.path.join(self.save_dir, f"step_metrics.json")
            with open(step_file, 'w') as f:
                json.dump(self._convert_for_json(self.step_metrics), f, indent=2)
            print(f"Step metrics (JSON) saved to: {step_file}")
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if not self.step_metrics:
            return {}
            
        rewards = [m['reward'] for m in self.step_metrics]
        energies = [m.get('energy_total', 0) for m in self.step_metrics]
        latencies = [m.get('latency_total_avg', 0) for m in self.step_metrics]
        qos_scores = [m.get('qos_quality_score_avg', 0) for m in self.step_metrics]
        
        return {
            'total_steps': len(self.step_metrics),
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'energy_stats': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies)
            },
            'latency_stats': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'qos_stats': {
                'mean': np.mean(qos_scores),
                'std': np.std(qos_scores),
                'min': np.min(qos_scores),
                'max': np.max(qos_scores)
            }
        }

def validate_environment(env, agent, state_dim, action_dim, device, epoch):
    """
    Validation function để in ra thông tin chi tiết về môi trường UAV
    """
    print(f"\n{'='*80}")
    print(f"VALIDATION AT EPOCH {epoch}")
    print(f"{'='*80}")
    
    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result
    
    # Test một action để xem phản hồi của môi trường và lấy info
    print(f"\n🎯 TESTING AGENT ACTION:")
    with torch.no_grad():
        action = agent.sample_action(state)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
            if action.ndim > 1:
                action = action[0]
    
    print(f"   Generated Action: {action}")
    
    # Convert action cho environment
    if hasattr(env.action_space, 'spaces'):
        uav_movement_dim = env.action_space['uav_movement'].shape[0]
        denoising_dim = env.action_space['denoising_steps'].shape[0]
        
        # Improve denoising steps conversion - ensure positive integers
        raw_denoising = action[uav_movement_dim:uav_movement_dim + denoising_dim]
        # Scale and clip to reasonable denoising steps (1-50)
        denoising_steps = np.clip(np.abs(raw_denoising * 25) + 1, 1, 50).astype(int)
        
        action_dict = {
            "uav_movement": action[:uav_movement_dim],
            "denoising_steps": denoising_steps
        }
        env_action = action_dict
        print(f"   UAV Movement: {action_dict['uav_movement']}")
        print(f"   Raw Denoising: {raw_denoising}")
        print(f"   Processed Denoising Steps: {action_dict['denoising_steps']}")
        
        # Calculate UAV movement magnitude
        movement_magnitude = np.linalg.norm(action_dict['uav_movement'])
        print(f"   UAV Movement Magnitude: {movement_magnitude:.4f}")
    else:
        env_action = action
        print(f"   Direct Action: {env_action}")
    
    # Execute action to get info
    step_result = env.step(env_action)
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        next_state, reward, done, info = step_result
    
    # Get current environment state
    print(f"\n📍 UAV POSITION:")
    if hasattr(env, 'uav_position'):
        print(f"   Current UAV Position: {env.uav_position}")
    if hasattr(env, 'uav_velocity'):
        print(f"   Current UAV Velocity: {env.uav_velocity}")
    if hasattr(env, 'flight_mode'):
        print(f"   Flight Mode: {env.flight_mode}")
    
    print(f"\n👥 USER EQUIPMENT (UE) STATUS:")
    if hasattr(env, 'users') and env.users:
        for i, user in enumerate(env.users):
            print(f"   UE {i+1}:")
            if hasattr(user, 'position'):
                print(f"      Position: {user.position}")
            if hasattr(user, 'task_queue'):
                print(f"      Task Queue Length: {len(user.task_queue) if user.task_queue else 0}")
            if hasattr(user, 'current_task'):
                task_type = getattr(user.current_task, 'type', 'None') if user.current_task else 'None'
                print(f"      Current Task: {task_type}")
            if hasattr(user, 'data_size'):
                print(f"      Data Size: {user.data_size}")
            if hasattr(user, 'computation_load'):
                print(f"      Computation Load: {user.computation_load}")
    else:
        # Try alternative attributes from the environment
        if hasattr(env, 'num_users'):
            print(f"   Total Users: {env.num_users}")
        if hasattr(env, 'user_positions'):
            print(f"   User Positions: {env.user_positions}")
        if hasattr(env, 'user_data_sizes'):
            print(f"   User Data Sizes: {env.user_data_sizes}")
        if hasattr(env, 'user_computation_loads'):
            print(f"   User Computation Loads: {env.user_computation_loads}")
        
        # Check from info if available (info is now available from step execution)
        if info and 'total_genai_ues' in info:
            print(f"   GenAI UEs: {info['total_genai_ues']}")
        if info and 'total_dnn_ues' in info:
            print(f"   DNN UEs: {info['total_dnn_ues']}")
    
    print(f"\n📡 BASE STATION STATUS:")
    if hasattr(env, 'BS_position'):
        print(f"   BS Position: {env.BS_position}")
    if hasattr(env, 'total_energy_consumption'):
        print(f"   Total Energy Consumption: {env.total_energy_consumption:.4f}")
    elif info and 'energy_total_consumption' in info:
        print(f"   Total Energy Consumption: {info['energy_total_consumption']:.4f}")
    if hasattr(env, 'total_latency'):
        print(f"   Total Latency: {env.total_latency:.4f}")
    elif info and 'latency_total_avg' in info:
        print(f"   Average Latency: {info['latency_total_avg']:.4f}")
    
    # Add UAV position if available
    if hasattr(env, 'uav_position'):
        print(f"   UAV Position: {env.uav_position}")
    elif info and 'uav_position' in info:
        print(f"   UAV Position: {info['uav_position']}")
    
    print(f"\n📊 STEP RESULT:")
    print(f"   Reward: {reward:.6f}")
    print(f"   Done: {done}")
    print(f"   State Dimension: {len(next_state)}")
    print(f"   State Range: [{np.min(next_state):.4f}, {np.max(next_state):.4f}]")
    
    if info:
        print(f"   Info Keys: {list(info.keys())}")
        for key, value in info.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value:.6f}")
            elif isinstance(value, (list, tuple, np.ndarray)):
                if len(value) < 10:  # Only print if not too long
                    print(f"      {key}: {value}")
                else:
                    print(f"      {key}: Array of length {len(value)}")
            else:
                print(f"      {key}: {value}")
    
    # Check for any completion status
    print(f"\n✅ TASK COMPLETION STATUS:")
    if hasattr(env, 'users') and env.users:
        completed_tasks = 0
        total_tasks = 0
        for i, user in enumerate(env.users):
            if hasattr(user, 'completed_tasks'):
                user_completed = len(user.completed_tasks) if user.completed_tasks else 0
                completed_tasks += user_completed
                print(f"   UE {i+1}: {user_completed} tasks completed")
            if hasattr(user, 'task_queue'):
                user_total = len(user.task_queue) if user.task_queue else 0
                total_tasks += user_total
                print(f"   UE {i+1}: {user_total} tasks in queue")
        
        if total_tasks > 0:
            completion_rate = completed_tasks / (completed_tasks + total_tasks) * 100
            print(f"   Overall Completion Rate: {completion_rate:.2f}%")
    else:
        # Use info data for completion analysis
        if info and 'completion_rate' in info:
            print(f"   Overall Completion Rate: {info['completion_rate']:.2f}%")
        if info and 'genai_completion_rate' in info:
            print(f"   GenAI Completion Rate: {info['genai_completion_rate']:.2f}%")
        if info and 'dnn_completion_rate' in info:
            print(f"   DNN Completion Rate: {info['dnn_completion_rate']:.2f}%")
        if info and 'ue_data_sent' in info:
            sent_count = np.sum(info['ue_data_sent']) if hasattr(info['ue_data_sent'], '__len__') else 0
            print(f"   UEs Data Sent: {sent_count}")
        if info and 'ue_data_processed' in info:
            processed_count = np.sum(info['ue_data_processed']) if hasattr(info['ue_data_processed'], '__len__') else 0
            print(f"   UEs Data Processed: {processed_count}")
        if info and 'ue_data_received' in info:
            received_count = np.sum(info['ue_data_received']) if hasattr(info['ue_data_received'], '__len__') else 0
            print(f"   UEs Data Received: {received_count}")
    
    # Performance analysis
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    if info and 'energy_efficiency_per_ue' in info:
        print(f"   Energy Efficiency per UE: {info['energy_efficiency_per_ue']:.6f}")
    if info and 'qos_quality_score_avg' in info:
        print(f"   Average QoS Quality Score: {info['qos_quality_score_avg']:.6f}")
    if info and 'qos_denoising_efficiency' in info:
        print(f"   Denoising Efficiency: {info['qos_denoising_efficiency']:.6f}")
    if info and 'latency_deadline_violations' in info:
        print(f"   Deadline Violations: {info['latency_deadline_violations']:.6f}")
    
    # Reward decomposition analysis
    print(f"\n🎯 REWARD ANALYSIS:")
    print(f"   Current Reward: {reward:.6f}")
    if reward > 0:
        print(f"   Reward Status: ✅ POSITIVE (Good performance)")
    else:
        print(f"   Reward Status: ❌ NEGATIVE (Poor performance)")
    
    # Try to estimate reward components
    energy_component = info.get('energy_total_consumption', 0) if info else 0
    latency_component = info.get('latency_total_avg', 0) if info else 0
    qos_component = info.get('qos_quality_score_avg', 0) if info else 0
    
    print(f"   Energy Component: -{energy_component:.6f}")
    print(f"   Latency Component: -{latency_component:.6f}")
    print(f"   QoS Component: +{qos_component:.6f}")
    
    print(f"{'='*80}\n")

hyperparameters = {
    'uav-genai-env': {
        'lr': 4e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
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
                      gamma=args.discount,
                      tau=0.99,
                      clip_param=0.2,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=5,  # More timesteps
                      lr=5e-3,  # Higher learning rate
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=1.0,
                      entropy_coef=0.04,  # More exploration
                      value_loss_coef=0.25)  # Less value weight
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
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(save_dir=os.path.join(output_dir, "metrics"))

    print_banner("Training Start", separator="*", num_star=90)
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        total_reward = 0
        episode_rewards = []
        epoch_losses = {}  # Collect losses for this epoch
        
        for episode in range(args.num_episodes_per_epoch):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            done = False
            ep_reward = 0
            step_count = 0
            
            while not done:
                action = agent.sample_action(state)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()[0]
                
                # Handle Dict action space
                if hasattr(env.action_space, 'spaces'):
                    # Convert flat action to Dict format
                    uav_movement_dim = env.action_space['uav_movement'].shape[0]
                    denoising_dim = env.action_space['denoising_steps'].shape[0]
                    
                    # Improve denoising steps conversion - ensure positive integers
                    raw_denoising = action[uav_movement_dim:uav_movement_dim + denoising_dim]
                    # Scale and clip to reasonable denoising steps (1-50)
                    denoising_steps = np.clip(np.abs(raw_denoising * 25) + 1, 1, 50).astype(int)
                    
                    action_dict = {
                        "uav_movement": action[:uav_movement_dim],
                        "denoising_steps": denoising_steps
                        # "denoising_steps": action[uav_movement_dim:uav_movement_dim + denoising_dim].astype(int)
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
                
                # Get agent losses if training occurs
                agent_losses = None
                if replay_buffer.size > args.batch_size:
                    loss_info = agent.train(replay_buffer, iterations=1, batch_size=args.batch_size)
                    if isinstance(loss_info, dict):
                        agent_losses = loss_info
                        # Accumulate losses for epoch
                        for loss_name, loss_value in agent_losses.items():
                            if loss_name not in epoch_losses:
                                epoch_losses[loss_name] = []
                            epoch_losses[loss_name].append(loss_value)
                    elif loss_info is not None:
                        agent_losses = {'agent_loss': loss_info}
                        if 'agent_loss' not in epoch_losses:
                            epoch_losses['agent_loss'] = []
                        epoch_losses['agent_loss'].append(loss_info)
                
                # Log step metrics with losses
                metrics_logger.log_step_metrics(
                    epoch=epoch,
                    episode=episode, 
                    step=step_count,
                    state=state,
                    action=env_action,
                    reward=reward,
                    info=info,
                    agent_losses=agent_losses
                )
                
                state = next_state
                ep_reward += reward
                step_count += 1
                
            episode_rewards.append(ep_reward)
            total_reward += ep_reward
            
        avg_reward = total_reward / args.num_episodes_per_epoch
        rewards.append(avg_reward)
        
        # Log epoch metrics
        agent_epoch_losses = None
        if epoch_losses:
            # Calculate average losses for the epoch
            agent_epoch_losses = {}
            for loss_name, loss_values in epoch_losses.items():
                if loss_values:
                    agent_epoch_losses[f"avg_{loss_name}"] = np.mean(loss_values)
                    agent_epoch_losses[f"std_{loss_name}"] = np.std(loss_values)
        
        reward_improved = metrics_logger.log_epoch_metrics(epoch, episode_rewards, agent_epoch_losses)
        
        # Auto-save plot at various intervals and milestones
        should_save_plot = False
        
        # Regular intervals
        if (epoch + 1) % 50 == 0:
            should_save_plot = True
            
        # Important milestones
        if (epoch + 1) in [10, 25, 100, 250, 500, 1000]:
            should_save_plot = True
            
        # When reward improves significantly (every 100 epochs check)
        if reward_improved and (epoch + 1) % 25 == 0:
            should_save_plot = True
            print(f"🎯 New best reward achieved: {avg_reward:.4f} at epoch {epoch + 1}")
            
        if should_save_plot:
            metrics_logger.save_loss_plot(save_interval=50)
        
        # Validation every 50 epochs
        if (epoch + 1) % 50 == 0:
            validate_environment(env, agent, state_dim, action_dim, device, epoch + 1)
        
        # More frequent logging for better tracking
        if (epoch + 1) % 5 == 0:
            logger.record_tabular('Epoch', epoch + 1)
            logger.record_tabular('Epoch Reward', avg_reward)
            logger.record_tabular('Avg Reward (last 10)', np.mean(rewards[-10:]))
            logger.record_tabular('Avg Reward (last 50)', np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards))
            logger.record_tabular('Best Reward', np.max(rewards))
            logger.record_tabular('Replay Buffer Size', replay_buffer.size)
            logger.dump_tabular()
        np.save(os.path.join(output_dir, "epoch_rewards.npy"), np.array(rewards))
        
        # Save metrics periodically
        if (epoch + 1) % 100 == 0:
            # metrics_logger.save_to_csv()
            metrics_logger.save_to_json()
    
    # Final validation
    print_banner("FINAL VALIDATION", separator="=", num_star=90)
    validate_environment(env, agent, state_dim, action_dim, device, args.num_epochs)
    
    # Save final metrics
    print_banner("SAVING FINAL METRICS", separator="=", num_star=90)
    metrics_logger.save_to_csv()
    metrics_logger.save_to_json()
    
    # Save final comprehensive plot
    final_plot_path = metrics_logger.save_final_plot()
    
    # Print summary statistics
    summary_stats = metrics_logger.get_summary_stats()
    print("\n📊 TRAINING SUMMARY STATISTICS:")
    print(f"   Total Training Steps: {summary_stats.get('total_steps', 0)}")
    
    if final_plot_path:
        print(f"📈 Final training plot saved to: {final_plot_path}")
    
    if 'reward_stats' in summary_stats:
        reward_stats = summary_stats['reward_stats']
        print(f"\n🎯 REWARD STATISTICS:")
        print(f"   Mean: {reward_stats['mean']:.6f}")
        print(f"   Std:  {reward_stats['std']:.6f}")
        print(f"   Min:  {reward_stats['min']:.6f}")
        print(f"   Max:  {reward_stats['max']:.6f}")
    
    if 'energy_stats' in summary_stats:
        energy_stats = summary_stats['energy_stats']
        print(f"\n⚡ ENERGY STATISTICS:")
        print(f"   Mean: {energy_stats['mean']:.6f}")
        print(f"   Std:  {energy_stats['std']:.6f}")
        print(f"   Min:  {energy_stats['min']:.6f}")
        print(f"   Max:  {energy_stats['max']:.6f}")
    
    if 'latency_stats' in summary_stats:
        latency_stats = summary_stats['latency_stats']
        print(f"\n⏱️ LATENCY STATISTICS:")
        print(f"   Mean: {latency_stats['mean']:.6f}")
        print(f"   Std:  {latency_stats['std']:.6f}")
        print(f"   Min:  {latency_stats['min']:.6f}")
        print(f"   Max:  {latency_stats['max']:.6f}")
    
    if 'qos_stats' in summary_stats:
        qos_stats = summary_stats['qos_stats']
        print(f"\n🎖️ QOS STATISTICS:")
        print(f"   Mean: {qos_stats['mean']:.6f}")
        print(f"   Std:  {qos_stats['std']:.6f}")
        print(f"   Min:  {qos_stats['min']:.6f}")
        print(f"   Max:  {qos_stats['max']:.6f}")
    
    print(f"\nTraining completed. Results saved to {output_dir}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Final average reward: {np.mean(rewards[-10:]):.4f}")
    print(f"Best average reward: {np.max(rewards):.4f}")
    print(f"Rewards trend: {rewards[-1] - rewards[0]:.4f} (final - initial)")
    
    return metrics_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='exp_1', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--dir", default="results", type=str)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--num_epochs", default=2000, type=int)
    parser.add_argument("--num_episodes_per_epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    # Algorithm choice
    parser.add_argument("--algo", default="dql", type=str, choices=['ppo', 'dql', 'gppo', 'gdql', 'a2c'])

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

    # Initial validation before training
    print_banner("INITIAL VALIDATION", separator="=", num_star=90)
    
    # Create a dummy agent for initial validation
    if args.algo == 'dql':
        from agents.ql_diffusion import Diffusion_QL as DummyAgent
        dummy_agent = DummyAgent(state_dim=state_dim,
                                 action_dim=action_dim, 
                                 max_action=max_action, 
                                 device=device,
                                 discount=args.discount,
                                 tau=args.tau
                                 )
    elif args.algo == 'ppo':
        from agents.ppo_diffusion import Diffusion_PPO as DummyAgent
        dummy_agent = DummyAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device, lr=0.0005)
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as DummyAgent
        dummy_agent = DummyAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    elif args.algo == 'gdql':
        from agents.gaussian_dql import Gaussian_DQL as DummyAgent
        dummy_agent = DummyAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    elif args.algo == 'a2c':
        from agents.a2c_agent import A2C_Agent as DummyAgent
        dummy_agent = DummyAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
    else:
        # Default random agent for initial validation
        class RandomAgent:
            def sample_action(self, state):
                return np.random.randn(action_dim)
        dummy_agent = RandomAgent()
    
    validate_environment(env, dummy_agent, state_dim, action_dim, device, 0)

    metrics_logger = train_agent(env, state_dim, action_dim, max_action, device, results_dir, args)
    
    print(f"\n🎉 All metrics saved to: {os.path.join(results_dir, 'metrics')}")
    print(f"   - CSV files for analysis in Excel/Python")
    print(f"   - JSON files for programmatic access")
    print(f"   - Detailed step-by-step and epoch-by-epoch data")
