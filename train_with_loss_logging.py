import numpy as np
import torch
import os
import json
from datetime import datetime
from agents.ql_diffusion import Diffusion_QL
from env import GAIServiceEnv

class LossLogger:
    def __init__(self, save_dir="loss_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize loss tracking
        self.losses = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
            'step': []
        }
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_losses(self, metrics, step, target_q_mean=None):
        """Log loss values for current step"""
        self.losses['step'].append(step)
        self.losses['bc_loss'].append(metrics['bc_loss'][-1])
        self.losses['ql_loss'].append(metrics['ql_loss'][-1])
        self.losses['actor_loss'].append(metrics['actor_loss'][-1])
        self.losses['critic_loss'].append(metrics['critic_loss'][-1])
        
        if target_q_mean is not None:
            self.losses['target_q_mean'].append(target_q_mean)
        
        # Print current losses
        if step % 100 == 0:  # Print every 100 steps
            print(f"Step {step}:")
            print(f"  BC Loss: {metrics['bc_loss'][-1]:.6f}")
            print(f"  QL Loss: {metrics['ql_loss'][-1]:.6f}")
            print(f"  Actor Loss: {metrics['actor_loss'][-1]:.6f}")
            print(f"  Critic Loss: {metrics['critic_loss'][-1]:.6f}")
            if target_q_mean is not None:
                print(f"  Target Q Mean: {target_q_mean:.6f}")
            print("-" * 50)
    
    def save_losses(self, filename=None):
        """Save all logged losses to file"""
        if filename is None:
            filename = f"diffusion_ql_losses_{self.timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for key, values in self.losses.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], np.ndarray):
                    save_data[key] = [v.tolist() if hasattr(v, 'tolist') else v for v in values]
                else:
                    save_data[key] = values
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Losses saved to: {filepath}")
        return filepath
    
    def plot_losses(self):
        """Plot the loss curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Diffusion QL Training Losses', fontsize=16)
            
            # BC Loss
            axes[0, 0].plot(self.losses['step'], self.losses['bc_loss'])
            axes[0, 0].set_title('BC Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # QL Loss
            axes[0, 1].plot(self.losses['step'], self.losses['ql_loss'])
            axes[0, 1].set_title('QL Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
            
            # Actor Loss
            axes[1, 0].plot(self.losses['step'], self.losses['actor_loss'])
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
            # Critic Loss
            axes[1, 1].plot(self.losses['step'], self.losses['critic_loss'])
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f"loss_plot_{self.timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to: {plot_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")

def train_diffusion_ql_with_logging():
    """Train Diffusion QL with detailed loss logging"""
    
    config = {
        "num_users": 10,
        "max_time": 100,
        "latency_limit": 5.0,
        "max_flops": 1e12,
        "max_vram": 8e9,
        "penalty_qos": 10.0,
        "penalty_latency": 5.0,
        "max_denoise_steps": 50,
        "lambda_qos": 10.0,
        "lambda_latency": 5.0,
        "lambda_mem": 1.0,
        "lambda_flops": 1.0,
        "T": 5,
        "tau": 5.0,
        "Gmax": 1e12,
        "Mmax": 8e9,
        "upload_rate": 10e6,
        "mem_rate": 10e6,
        "compute_power": 1e9,
        "download_rate": 10e6,
        "base_fid": 50,
        "base_price": 0.1,
        "base_price_mem": 0.05,
        "base_price_flops": 0.00001,
        "base_price_latency": 0.00001,
        "base_price_qos": 0.00001,
        "base_price_mem": 0.00001,
        "base_price_flops": 0.00001,
        "base_price_latency": 0.00001,
        "base_price_qos": 0.00001,
        
    }
    env = GAIServiceEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = Diffusion_QL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        lr=3e-4
    )
    
    # Initialize loss logger
    logger = LossLogger()
    
    # Training parameters
    total_steps = 10000
    batch_size = 64
    iterations_per_step = 1
    
    # Create replay buffer (for online RL, you'd collect data interactively)
    from main import ReplayBuffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000, device=device)
    # Collect some initial data
    print("Collecting initial data...")
    for _ in range(1000):
        state = env.reset()
        for _ in range(100):  # Collect 100 steps per episode
            action = env.action_space.sample()  # Random actions for initial data
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if done:
                break
    
    
    # Training loop
    print("Starting training...")
    for step in range(total_steps):
        # Train agent
        metrics = agent.train(replay_buffer, iterations_per_step, batch_size)
        
        # Log losses
        logger.log_losses(metrics, step)
        
        # Save losses periodically
        if step % 1000 == 0 and step > 0:
            logger.save_losses()
    
    # Final save and plot
    logger.save_losses("final_losses.json")
    logger.plot_losses()
    
    print("Training completed!")
    return agent, logger

if __name__ == "__main__":
    agent, logger = train_diffusion_ql_with_logging() 