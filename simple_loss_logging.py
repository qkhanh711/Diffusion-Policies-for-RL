import numpy as np
import torch
from agents.ql_diffusion import Diffusion_QL
from env import GAIServiceEnv

def print_losses(metrics, step):
    """Print loss values for current step"""
    print(f"Step {step:5d} | "
          f"BC: {metrics['bc_loss'][-1]:8.4f} | "
          f"QL: {metrics['ql_loss'][-1]:8.4f} | "
          f"Actor: {metrics['actor_loss'][-1]:8.4f} | "
          f"Critic: {metrics['critic_loss'][-1]:8.4f}")

def train_with_simple_logging():
    """Train Diffusion QL with simple loss printing"""
    
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
    
    # Training parameters
    total_steps = 1000  # Reduced for demo
    batch_size = 64
    iterations_per_step = 1
    
    # Create replay buffer
    from main import ReplayBuffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, max_size=int(1e6))
    
    # Collect initial data
    print("Collecting initial data...")
    for _ in range(100):
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
    print("Step     | BC Loss  | QL Loss  | Actor    | Critic")
    print("-" * 55)
    
    # Training loop
    for step in range(total_steps):
        # Train agent
        metrics = agent.train(replay_buffer, iterations_per_step, batch_size)
        
        # Print losses every 50 steps
        if step % 50 == 0:
            print_losses(metrics, step)
    
    print("\nTraining completed!")
    return agent

if __name__ == "__main__":
    agent = train_with_simple_logging() 