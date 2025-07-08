import numpy as np
import torch
from agents.ql_diffusion import Diffusion_QL
from env import GAIServiceEnv

def print_loss_values():
    """Print loss values from a single training step"""
    
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
    
    # Create replay buffer
    from utils.replay_buffer import ReplayBuffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, max_size=int(1e6))
    
    # Collect some data
    print("Collecting data...")
    for _ in range(100):
        state = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            if done:
                break
    
    print(f"Data collected: {len(replay_buffer)} transitions")
    
    # Train for a few steps and print losses
    print("\nTraining and printing losses...")
    print("Step | BC Loss  | QL Loss  | Actor Loss | Critic Loss")
    print("-" * 55)
    
    for step in range(10):  # Just 10 steps for demo
        metrics = agent.train(replay_buffer, iterations=1, batch_size=64)
        
        print(f"{step:4d} | "
              f"{metrics['bc_loss'][-1]:8.4f} | "
              f"{metrics['ql_loss'][-1]:8.4f} | "
              f"{metrics['actor_loss'][-1]:10.4f} | "
              f"{metrics['critic_loss'][-1]:10.4f}")
    
    print("\nLoss breakdown:")
    print(f"BC Loss (Behavior Cloning): {metrics['bc_loss'][-1]:.6f}")
    print(f"QL Loss (Q-Learning): {metrics['ql_loss'][-1]:.6f}")
    print(f"Actor Loss (Total): {metrics['actor_loss'][-1]:.6f}")
    print(f"Critic Loss: {metrics['critic_loss'][-1]:.6f}")
    
    return agent

if __name__ == "__main__":
    agent = print_loss_values() 