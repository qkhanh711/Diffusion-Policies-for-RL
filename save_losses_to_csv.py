import numpy as np
import torch
import pandas as pd
import os
from datetime import datetime
from agents.ql_diffusion import Diffusion_QL
from env import GAIServiceEnv

class CSVLossLogger:
    def __init__(self, save_dir="loss_csv"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data storage
        self.loss_data = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_losses(self, metrics, step, additional_info=None):
        """Log loss values for current step"""
        loss_dict = {
            'step': step,
            'bc_loss': metrics['bc_loss'][-1],
            'ql_loss': metrics['ql_loss'][-1],
            'actor_loss': metrics['actor_loss'][-1],
            'critic_loss': metrics['critic_loss'][-1]
        }
        
        # Add additional info if provided
        if additional_info:
            loss_dict.update(additional_info)
        
        self.loss_data.append(loss_dict)
        
        # Print every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: BC={loss_dict['bc_loss']:.4f}, "
                  f"QL={loss_dict['ql_loss']:.4f}, "
                  f"Actor={loss_dict['actor_loss']:.4f}, "
                  f"Critic={loss_dict['critic_loss']:.4f}")
    
    def save_to_csv(self, filename=None):
        """Save all logged losses to CSV file"""
        if filename is None:
            filename = f"diffusion_ql_losses_{self.timestamp}.csv"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.loss_data)
        df.to_csv(filepath, index=False)
        
        print(f"Losses saved to CSV: {filepath}")
        print(f"Total steps logged: {len(self.loss_data)}")
        
        # Print summary statistics
        print("\nLoss Summary Statistics:")
        print(df.describe())
        
        return filepath
    
    def plot_losses(self):
        """Plot loss curves using pandas"""
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.loss_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Diffusion QL Training Losses', fontsize=16)
            
            # BC Loss
            axes[0, 0].plot(df['step'], df['bc_loss'])
            axes[0, 0].set_title('BC Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # QL Loss
            axes[0, 1].plot(df['step'], df['ql_loss'])
            axes[0, 1].set_title('QL Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
            
            # Actor Loss
            axes[1, 0].plot(df['step'], df['actor_loss'])
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
            
            # Critic Loss
            axes[1, 1].plot(df['step'], df['critic_loss'])
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

def train_and_save_losses():
    """Train Diffusion QL and save losses to CSV"""
    
    # Environment setup
    env = GAIServiceEnv()
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
    
    # Initialize CSV logger
    logger = CSVLossLogger()
    
    # Training parameters
    total_steps = 2000
    batch_size = 64
    iterations_per_step = 1
    
    # Create replay buffer
    from utils.replay_buffer import ReplayBuffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, max_size=int(1e6))
    
    # Collect initial data
    print("Collecting initial data...")
    for _ in range(200):
        state = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if done:
                break
    
    print(f"Initial data: {len(replay_buffer)} transitions")
    print("\nStarting training...")
    
    # Training loop
    for step in range(total_steps):
        # Train agent
        metrics = agent.train(replay_buffer, iterations_per_step, batch_size)
        
        # Log losses
        logger.log_losses(metrics, step)
        
        # Save periodically
        if step % 500 == 0 and step > 0:
            logger.save_to_csv(f"intermediate_losses_step_{step}.csv")
    
    # Final save
    logger.save_to_csv("final_losses.csv")
    logger.plot_losses()
    
    print("Training completed!")
    return agent, logger

if __name__ == "__main__":
    agent, logger = train_and_save_losses() 