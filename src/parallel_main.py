#!/usr/bin/env python3
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


class PlottingUtilities:
    """
    Tách riêng các chức năng plotting từ MetricsLogger gốc
    Sử dụng data từ env.metrics_logger để tạo plots và GIFs
    """
    
    def __init__(self, save_dir="plots", timestamp=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🎨 PlottingUtilities initialized: {save_dir}")
    
    def create_training_progress_plot(self, metrics_logger):
        """Tạo plot tiến trình training từ metrics_logger của env"""
        try:
            episode_stats = metrics_logger.get_episode_statistics()
            if not episode_stats or episode_stats['total_episodes'] == 0:
                print("⚠️ No episode data available for plotting")
                return None
            
            # Lấy data từ episode rewards
            rewards = list(metrics_logger.episode_rewards)
            episodes = list(range(1, len(rewards) + 1))
            
            if len(rewards) == 0:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress Summary', fontsize=16)
            
            # Plot 1: Episode Rewards
            axes[0, 0].plot(episodes, rewards, 'b-', linewidth=1, alpha=0.7, label='Episode Reward')
            if len(rewards) > 10:
                window_size = min(20, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                moving_episodes = episodes[window_size-1:]
                axes[0, 0].plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
            
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Training Rewards')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Add statistics
            if len(rewards) > 0:
                stats_text = f'Final: {rewards[-1]:.2f}\\nBest: {max(rewards):.2f}\\nAvg: {np.mean(rewards):.2f}'
                axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Service Ratios
            service_ratios = list(metrics_logger.episode_service_ratios)
            if service_ratios:
                axes[0, 1].plot(episodes, service_ratios, 'g-', linewidth=2, marker='o', markersize=3)
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Service Ratio')
                axes[0, 1].set_title('Service Efficiency')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_ylim(0, 1)
            
            # Plot 3: QoS Performance (if available)
            if len(metrics_logger.step_metrics['avg_qos_achieved']) > 0:
                qos_data = list(metrics_logger.step_metrics['avg_qos_achieved'])
                qos_satisfaction = list(metrics_logger.step_metrics['qos_satisfaction_rate'])
                step_nums = list(range(1, len(qos_data) + 1))
                
                axes[1, 0].plot(step_nums, qos_data, 'purple', linewidth=2, label='QoS Achieved')
                axes[1, 0].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target (30)')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('QoS Score')
                axes[1, 0].set_title('Quality of Service')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                
                # Twin axis for satisfaction rate
                ax_twin = axes[1, 0].twinx()
                ax_twin.plot(step_nums, qos_satisfaction, 'orange', linewidth=2, alpha=0.7, label='Satisfaction Rate')
                ax_twin.set_ylabel('Satisfaction Rate')
                ax_twin.set_ylim(0, 1)
            
            # Plot 4: Diffusion Steps (if available)
            if len(metrics_logger.step_metrics['avg_diffusion_steps']) > 0:
                diff_steps = list(metrics_logger.step_metrics['avg_diffusion_steps'])
                min_steps = list(metrics_logger.step_metrics['min_diffusion_steps'])
                max_steps = list(metrics_logger.step_metrics['max_diffusion_steps'])
                step_nums = list(range(1, len(diff_steps) + 1))
                
                axes[1, 1].plot(step_nums, diff_steps, 'brown', linewidth=2, label='Average')
                axes[1, 1].fill_between(step_nums, min_steps, max_steps, alpha=0.3, label='Range')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Diffusion Steps')
                axes[1, 1].set_title('Diffusion Steps Usage')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f"training_progress_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Training progress plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Failed to create training progress plot: {e}")
            return None
    
    def create_resource_utilization_plot(self, metrics_logger):
        """Tạo plot về việc sử dụng tài nguyên"""
        try:
            if len(metrics_logger.step_metrics['memory_utilization']) == 0:
                return None
            
            # Extract resource utilization data
            memory_util = list(metrics_logger.step_metrics['memory_utilization'])
            peak_memory = list(metrics_logger.step_metrics['peak_memory'])
            total_flops = list(metrics_logger.step_metrics['total_flops'])
            time_spent = list(metrics_logger.step_metrics['time_spent'])
            num_batches = list(metrics_logger.step_metrics['num_batches'])
            
            step_nums = list(range(1, len(memory_util) + 1))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Resource Utilization Analysis', fontsize=16)
            
            # Plot 1: Memory Utilization
            axes[0, 0].plot(step_nums, memory_util, 'r-', linewidth=2, label='Memory Utilization')
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% Limit')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Utilization Rate')
            axes[0, 0].set_title('Memory Utilization')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, max(1.2, max(memory_util) * 1.1))
            
            # Plot 2: Peak Memory Usage
            axes[0, 1].plot(step_nums, peak_memory, 'orange', linewidth=2, label='Peak Memory (GB)')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Memory (GB)')
            axes[0, 1].set_title('Peak Memory Usage')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Plot 3: FLOPS Usage
            axes[1, 0].plot(step_nums, total_flops, 'purple', linewidth=2, label='Total FLOPS')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('FLOPS')
            axes[1, 0].set_title('Computational Load')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Plot 4: Batch Processing
            axes[1, 1].plot(step_nums, num_batches, 'green', linewidth=2, marker='o', markersize=3, label='Number of Batches')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Number of Batches')
            axes[1, 1].set_title('Batch Processing Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f"resource_utilization_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔧 Resource utilization plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Failed to create resource utilization plot: {e}")
            return None
    
    def create_qos_analysis_plot(self, metrics_logger):
        """Tạo plot phân tích QoS chi tiết"""
        try:
            if len(metrics_logger.step_metrics['avg_qos_achieved']) == 0:
                return None
            
            qos_achieved = list(metrics_logger.step_metrics['avg_qos_achieved'])
            qos_required = list(metrics_logger.step_metrics['avg_qos_required'])
            qos_satisfaction = list(metrics_logger.step_metrics['qos_satisfaction_rate'])
            step_nums = list(range(1, len(qos_achieved) + 1))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Quality of Service Analysis', fontsize=16)
            
            # Plot 1: QoS Achieved vs Required
            axes[0, 0].plot(step_nums, qos_achieved, 'blue', linewidth=2, label='QoS Achieved')
            axes[0, 0].plot(step_nums, qos_required, 'red', linewidth=2, label='QoS Required')
            axes[0, 0].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Target (30)')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('QoS Score (BRISQUE)')
            axes[0, 0].set_title('QoS Performance (Lower = Better)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Plot 2: QoS Satisfaction Rate
            axes[0, 1].plot(step_nums, qos_satisfaction, 'green', linewidth=2, marker='o', markersize=3)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Satisfaction Rate')
            axes[0, 1].set_title('QoS Satisfaction Rate')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # Plot 3: QoS Gap Analysis
            qos_gap = [achieved - required for achieved, required in zip(qos_achieved, qos_required)]
            axes[1, 0].plot(step_nums, qos_gap, 'purple', linewidth=2, label='QoS Gap (Achieved - Required)')
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Perfect Match')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('QoS Gap')
            axes[1, 0].set_title('QoS Gap Analysis (Negative = Better)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Plot 4: QoS vs Diffusion Steps (if available)
            if len(metrics_logger.step_metrics['avg_diffusion_steps']) > 0:
                diffusion_steps = list(metrics_logger.step_metrics['avg_diffusion_steps'])
                # Create scatter plot
                axes[1, 1].scatter(diffusion_steps, qos_achieved, alpha=0.6, c=step_nums, cmap='viridis')
                axes[1, 1].set_xlabel('Average Diffusion Steps')
                axes[1, 1].set_ylabel('QoS Achieved')
                axes[1, 1].set_title('QoS vs Diffusion Steps Relationship')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add colorbar for time progression
                cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
                cbar.set_label('Step Number')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f"qos_analysis_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🎯 QoS analysis plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Failed to create QoS analysis plot: {e}")
            return None
    
    def create_comprehensive_summary_plot(self, metrics_logger):
        """Tạo plot tổng hợp toàn bộ metrics"""
        try:
            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35, 
                                top=0.95, bottom=0.08, left=0.06, right=0.94)
            
            # Episode Rewards (large plot)
            ax1 = fig.add_subplot(gs[0, :2])
            rewards = list(metrics_logger.episode_rewards)
            episodes = list(range(1, len(rewards) + 1))
            if rewards:
                ax1.plot(episodes, rewards, 'b-', linewidth=1, alpha=0.7)
                if len(rewards) > 10:
                    window = min(20, len(rewards) // 5)
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
                ax1.set_title('Training Progress - Episode Rewards', fontsize=12, pad=10)
                ax1.set_xlabel('Episode', fontsize=10)
                ax1.set_ylabel('Reward', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=9)
            
            # Service Performance
            ax2 = fig.add_subplot(gs[0, 2])
            service_ratios = list(metrics_logger.episode_service_ratios)
            if service_ratios:
                ax2.plot(episodes, service_ratios, 'g-', linewidth=2)
                ax2.set_title('Service Efficiency', fontsize=12, pad=10)
                ax2.set_xlabel('Episode', fontsize=10)
                ax2.set_ylabel('Service Ratio', fontsize=10)
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(labelsize=9)
            
            # Performance Metrics Summary
            ax3 = fig.add_subplot(gs[0, 3])
            
            # Get latest metrics data
            step_stats = metrics_logger.get_step_statistics()
            
            # Extract metrics
            peak_memory = step_stats.get('peak_memory', {}).get('mean', 0)  # GB
            avg_time_spent = step_stats.get('time_spent', {}).get('mean', 0) * 1000  # convert to ms
            mean_qos = step_stats.get('avg_qos_achieved', {}).get('mean', 0)
            
            # Create bar chart
            metrics_labels = ['Peak Memory\n(GB)', 'Latency\n(ms)', 'Mean QoS\n(BRISQUE)']
            metrics_values = [peak_memory, avg_time_spent, mean_qos]
            colors = ['steelblue', 'darkorange', 'forestgreen']
            
            bars = ax3.bar(metrics_labels, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax3.set_title('Key Performance Metrics', fontsize=12, pad=10)
            ax3.set_ylabel('Value', fontsize=10)
            ax3.tick_params(axis='x', labelsize=9)
            ax3.tick_params(axis='y', labelsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # QoS Performance
            if len(metrics_logger.step_metrics['avg_qos_achieved']) > 0:
                ax4 = fig.add_subplot(gs[1, :2])
                qos_achieved = list(metrics_logger.step_metrics['avg_qos_achieved'])
                qos_satisfaction = list(metrics_logger.step_metrics['qos_satisfaction_rate'])
                step_nums = list(range(1, len(qos_achieved) + 1))
                
                ax4.plot(step_nums, qos_achieved, 'purple', linewidth=2, label='QoS Achieved')
                ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target')
                ax4.set_title('QoS Performance Over Time', fontsize=12, pad=10)
                ax4.set_xlabel('Step', fontsize=10)
                ax4.set_ylabel('QoS Score', fontsize=10)
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3)
                ax4.tick_params(labelsize=9)
                
                # Twin axis for satisfaction
                ax4_twin = ax4.twinx()
                ax4_twin.plot(step_nums, qos_satisfaction, 'orange', linewidth=2, alpha=0.7)
                ax4_twin.set_ylabel('Satisfaction Rate', fontsize=10)
                ax4_twin.set_ylim(0, 1)
                ax4_twin.tick_params(labelsize=9)
            
            # Resource Utilization
            if len(metrics_logger.step_metrics['memory_utilization']) > 0:
                ax5 = fig.add_subplot(gs[1, 2])
                memory_util = list(metrics_logger.step_metrics['memory_utilization'])
                step_nums = list(range(1, len(memory_util) + 1))
                ax5.plot(step_nums, memory_util, 'red', linewidth=2)
                ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                ax5.set_title('Memory Utilization', fontsize=12, pad=10)
                ax5.set_xlabel('Step', fontsize=10)
                ax5.set_ylabel('Utilization Rate', fontsize=10)
                ax5.grid(True, alpha=0.3)
                ax5.tick_params(labelsize=9)
            
            # Diffusion Steps
            if len(metrics_logger.step_metrics['avg_diffusion_steps']) > 0:
                ax6 = fig.add_subplot(gs[1, 3])
                diff_steps = list(metrics_logger.step_metrics['avg_diffusion_steps'])
                step_nums = list(range(1, len(diff_steps) + 1))
                ax6.plot(step_nums, diff_steps, 'brown', linewidth=2)
                ax6.set_title('Diffusion Steps Usage', fontsize=12, pad=10)
                ax6.set_xlabel('Step', fontsize=10)
                ax6.set_ylabel('Average Steps', fontsize=10)
                ax6.grid(True, alpha=0.3)
                ax6.tick_params(labelsize=9)
            
            # Performance Summary (bottom row)
            ax7 = fig.add_subplot(gs[2, :])
            
            # Create summary statistics table
            episode_stats = metrics_logger.get_episode_statistics()
            step_stats = metrics_logger.get_step_statistics()
            violation_stats = metrics_logger.get_violation_statistics()
            
            summary_text = f"""TRAINING SUMMARY:
                                • Total Episodes: {episode_stats.get('total_episodes', 0)}
                                • Average Reward: {episode_stats.get('avg_episode_reward', 0):.2f} ± {episode_stats.get('std_episode_reward', 0):.2f}
                                • Best Reward: {episode_stats.get('max_episode_reward', 0):.2f}
                                • Average Service Ratio: {episode_stats.get('avg_service_ratio', 0):.3f}
                                
                                QoS PERFORMANCE:
                                • Average QoS Achieved: {step_stats.get('avg_qos_achieved', {}).get('mean', 0):.2f}
                                • QoS Satisfaction Rate: {step_stats.get('qos_satisfaction_rate', {}).get('mean', 0):.1%}
                                • Average Diffusion Steps: {step_stats.get('avg_diffusion_steps', {}).get('mean', 0):.1f}
                                
                                RESOURCE EFFICIENCY:
                                • Memory Utilization: {step_stats.get('memory_utilization', {}).get('mean', 0):.1%}
                                • Constraint Violations: {violation_stats['total_violations']} total
                                • Memory Violation Rate: {violation_stats['memory_violation_rate']:.1%}
                                • QoS Violation Rate: {violation_stats['qos_violation_rate']:.1%}"""
            
            ax7.text(0.05, 0.92, summary_text, transform=ax7.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.axis('off')
            ax7.set_title('Training Summary Statistics', fontsize=14, fontweight='bold', pad=20)
            
            plt.suptitle('Comprehensive Training Analysis', fontsize=20, fontweight='bold', y=0.98)
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f"comprehensive_summary_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"📊 Comprehensive summary plot saved: {plot_path}")
            return plot_path
            
        except Exception as e:
            print(f"⚠️ Failed to create comprehensive summary plot: {e}")
            return None
    
    def generate_all_plots(self, metrics_logger):
        """Tạo tất cả plots từ metrics_logger"""
        print(f"🎨 Generating all plots from metrics data...")
        
        plots_created = []
        
        # Training progress
        plot1 = self.create_training_progress_plot(metrics_logger)
        if plot1:
            plots_created.append(plot1)
        
        # Resource utilization
        plot2 = self.create_resource_utilization_plot(metrics_logger)
        if plot2:
            plots_created.append(plot2)
        
        # QoS analysis
        plot3 = self.create_qos_analysis_plot(metrics_logger)
        if plot3:
            plots_created.append(plot3)
        
        # Comprehensive summary
        plot4 = self.create_comprehensive_summary_plot(metrics_logger)
        if plot4:
            plots_created.append(plot4)
        
        print(f"✅ Generated {len(plots_created)} plots")
        return plots_created


# Keep existing hyperparameters và ReplayBuffer classes
hyperparameters = {
    'gail-service-env'       :  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v1'    :  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v3-org':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v4'    :  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v5'    :  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v6'    :  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1},
    'gail-service-env-v6-baseline':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1, 'gn': 5.0,  'top_k': 1}
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
    """
    Hàm training agent được đơn giản hóa, tận dụng MetricsLogger từ env
    """
    
    # Agent initialization (giữ nguyên)
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
        from agents.ppo_diffusion_fixed import Diffusion_PPO_Fixed as Agent
        # from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      lr=3.6e-4,  # Much lower LR to prevent collapse
                    #   gamma=args.discount,
                    #   tau=args.tau,
                    #   clip_param=0.1,  # Tight clipping for stability
                    #   beta_schedule=args.beta_schedule,
                    #   n_timesteps=args.T,
                    #   ema_decay=0.9999,  # Very slow EMA 
                    #   step_start_ema=5000,  # Very late EMA start
                    #   update_ema_every=50,  # Very infrequent updates
                    #   lr_decay=False,  # Disable LR decay for stability
                    #   lr_maxt=args.num_epochs,
                    #   grad_norm=0.25,  # Very strong gradient clipping
                    #   entropy_coef=0.0001,  # Minimal entropy for stability
                    #   value_loss_coef=0.1,  # Lower value loss to prevent overfitting
                    #   warmup_steps=100
                      )  # Very long warmup
    elif args.algo == 'dppo_v1':
        from agents.ppo_diffusion_v1 import Diffusion_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device)
    elif args.algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=0.0003,
            noise_scale=0.1,
            clip_ratio=0.1,
            value_clip_ratio=0.1,
            ent_coef=0.02,
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
    elif args.algo == 'a2c':
        from agents.gaussian_a2c import Gaussian_A2C as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device)
    elif args.algo == 'da2c':
        from agents.a2c_diffusion import Diffusion_A2C as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    # Initialize plotting utilities
    plot_dir = os.path.join(output_dir, "plots")
    plotter = PlottingUtilities(save_dir=plot_dir)
    
    # Initialize replay buffer và tracking
    evaluations = []
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    
    episode_rewards = []
    current_state = env.reset()

    for episode in tqdm(range(args.num_episodes), desc="Training Episodes"):
        state = env.reset()
        current_state = state
        done = False
        episode_reward = 0
        episode_losses = {}
        
        step_count = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Print detailed info if enabled
            if args.print_logger:
                _print_step_info(episode, step_count, info, config, reward)
            
            # Add to replay buffer
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            current_state = state
            episode_reward += reward
            step_count += 1

            # Train agent
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
        
        # Episode completed - print summary if enabled
        if args.print_logger:
            _print_episode_summary(episode, episode_reward, step_count, info, config)
        
        episode_rewards.append(episode_reward)
        
        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(
            agent, args.env_name, args.seed, eval_episodes=args.eval_episodes, 
            config=config, eval_env=env)
        
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, episode])
        
        np.save(os.path.join(output_dir, f"episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
        # Periodic saving và plotting
        if episode % args.save_episodes == 0:
            # Generate plots using env's metrics_logger
            if hasattr(env, 'metrics_logger') and env.metrics_logger:
                print(f"📊 Generating plots for episode {episode}...")
                plotter.generate_all_plots(env.metrics_logger)
                
                # Export detailed metrics
                env.export_metrics(os.path.join(output_dir, f"detailed_metrics.json"))
                
                # Print metrics summary
                print(f"\\n{'='*80}")
                print(f"METRICS SUMMARY - Episode {episode}")
                print(f"{'='*80}")
                env.get_metrics_summary(detailed=True)
            
            # Save model if needed
            if args.save_best_model:
                agent.save_model(output_dir, episode)
        
        # Logging with logger
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.record_tabular('Avg Reward (last 10)', np.mean(episode_rewards[-10:]))
        logger.record_tabular('Avg Reward (last 50)', np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards))
        logger.record_tabular('Best Reward', np.max(episode_rewards))
        logger.dump_tabular()

    # Training completed - final processing
    utils.print_banner(f"Training Complete", separator="*", num_star=90)
    
    # Save final results
    scores = np.array(evaluations)
    _save_best_scores(scores, output_dir, args)
    
    # Save final arrays
    if not os.path.exists(os.path.join(output_dir, "main")):
        os.makedirs(os.path.join(output_dir, "main"))
    np.save(os.path.join(output_dir, f"main/episode_rewards_{args.algo}.npy"), np.array(episode_rewards))
    
    # Final plots và metrics export
    if hasattr(env, 'metrics_logger') and env.metrics_logger:
        print(f"📊 Generating final comprehensive plots...")
        final_plots = plotter.generate_all_plots(env.metrics_logger)
        
        # Export final detailed metrics
        env.export_metrics(os.path.join(output_dir, "final_detailed_metrics.json"))
        
        # Print final summary
        print(f"\\n{'='*80}")
        print(f"FINAL TRAINING SUMMARY")
        print(f"{'='*80}")
        env.get_metrics_summary(detailed=True)
        
        print(f"\\n📁 All plots saved to: {plot_dir}")
        print(f"📊 Generated {len(final_plots)} final plots")
    
    print(f"✅ Training completed successfully!")


def _print_step_info(episode, step_count, info, config, reward):
    """Helper function để in thông tin step"""
    peak_mem = info.get('peak_mem', 0)
    mem_max = config.get('Mmax', 0)
    memory_util = info.get('memory_utilization', 0)
    num_batches = info.get('num_batches', 0)
    served = info.get('served', 0)
    total_users = config.get('num_users', 0)
    avg_qos = info.get('avg_qos_achieved', 0)
    avg_diff_steps = info.get('avg_diffusion_steps', 0)
    
    print(f"Episode {episode}, Step {step_count}: Memory = {peak_mem:.2f}/{mem_max:.2f} GB, "
          f"Usage = {memory_util*100:.1f}%, Batches = {num_batches}, "
          f"Served = {served}/{total_users}, QoS = {avg_qos:.1f}, "
          f"DiffSteps = {avg_diff_steps:.1f}, Reward = {reward:.4f}")


def _print_episode_summary(episode, episode_reward, step_count, info, config):
    """Helper function để in tóm tắt episode"""
    served = info.get('served', 0)
    service_ratio = info.get('service_ratio', 0)
    qos_achieved = info.get('avg_qos_achieved', 0)
    qos_satisfaction = info.get('qos_satisfaction_rate', 0)
    diffusion_steps = info.get('avg_diffusion_steps', 0)
    peak_memory = info.get('peak_mem', 0)
    memory_util = info.get('memory_utilization', 0)
    num_batches = info.get('num_batches', 0)
    total_flops = info.get('total_flops', 0)
    time_spent = info.get('time_spent', 0)
    
    print(f"\\n{'='*80}")
    print(f"EPISODE {episode} SUMMARY:")
    print(f"Episode Reward: {episode_reward:.4f}")
    print(f"Steps: {step_count}")
    print(f"Users Served: {served}/{config.get('num_users', 0)} ({service_ratio*100:.1f}%)")
    print(f"QoS: Achieved={qos_achieved:.2f}, Satisfaction={qos_satisfaction:.1%}")
    print(f"Diffusion Steps: Avg={diffusion_steps:.1f}")
    print(f"Resource Usage:")
    print(f"  - Peak Memory: {peak_memory:.2f}/{config.get('Mmax', 0):.1f} GB ({memory_util*100:.1f}%)")
    print(f"  - Total FLOPs: {total_flops:.2e}")
    print(f"  - Time Spent: {time_spent:.3f}s")
    print(f"  - Batches: {num_batches}")
    print(f"{'='*80}\\n")


def _save_best_scores(scores, output_dir, args):
    """Helper function để lưu best scores"""
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


def eval_policy(policy, env_name, seed, eval_episodes=10, config=None, eval_env=None):
    """Evaluation function (giữ nguyên)"""
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
    parser.add_argument("--env_name", default="gail-service-env-v6-baseline", type=str)
    parser.add_argument("--dir", default="parallel_results", type=str)                
    parser.add_argument("--seed", default=1, type=int)                      
    parser.add_argument("--num_episodes", default=100, type=int) 
    parser.add_argument("--save_episodes", default=20, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='linear', type=str)
    parser.add_argument("--algo", default="gppo", type=str)  
    parser.add_argument("--ms", default='online', type=str, help="['online', 'offline']")
    parser.add_argument("--n_users", default=10, type=int, help="Number of users in the environment")
    parser.add_argument("--print_logger", action='store_true', help="Print detailed resource usage to terminal")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    # Load hyperparameters
    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 50  # Reduced for faster evaluation
    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup logging directory
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
    
    # Save experiment configuration
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL-Parallel")

    # Setup environment
    if args.env_name == 'gail-service-env-v6-baseline':
        from env.env_v6_baseline import GAIServiceEnv_v1_baseline as GAIServiceEnv, EnvConfig_v1_baseline as EnvConfig
    elif args.env_name == 'gail-service-env-v6':
        from env.env_v6 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v5':
        from env.env_v5 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v4':
        from env.env_v4 import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v3-org':
        from env.env_v3_org import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env-v1':    
        from env.env import GAIServiceEnv_v1 as GAIServiceEnv, EnvConfig_v1 as EnvConfig
    elif args.env_name == 'gail-service-env':
        from env.gai_env_org import GAIServiceEnv, EnvConfig
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")
        
    config = EnvConfig(args.env_name)
    config["num_users"] = args.n_users

    print(f"Training config: {config}")
    
    # Create environment with MetricsLogger enabled
    if 'baseline' in args.env_name:
        env = GAIServiceEnv(config, enable_metrics=True, 
                           metrics_window_size=500,
                           log_file=os.path.join(results_dir, "episode_metrics.log"))
    else:
        env = GAIServiceEnv(config)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    
    # Setup logger
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    # Start training
    train_agent(env, state_dim, action_dim, max_action, args.device, results_dir, args)
