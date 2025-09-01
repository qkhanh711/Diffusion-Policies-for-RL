#!/usr/bin/env python3
"""
Test script for Gaussian PPO agent
"""

import argparse
import numpy as np
import os
import torch

from utils import utils
from utils.logger import logger, setup_logger
from uav_env import UAVGenAIEnv
from agents.gaussian_ppo import Gaussian_PPO

def get_uav_config():
    """Get configuration for UAVGenAIEnv"""
    return {
        "num_users": 3,  # Smaller for testing
        "Dmax": 10,
        "tau": 3.0,
        "area": [-100, 100, -100, 100],
        "z_range": [50, 200],
        "BS_position": [0, 0, 50],
        "phi": 1.5e-9,
        "f_inf_BS": 1.0e9,
        "Cin_BS": 0.03125,
        "lambda_Q": 0.5,
        "lambda_L": 0.5,
        "lambda_E": 1.0,
        "psi": 10.0,
        "Qreq": [30, 30, 30],
        "T": 5
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

def test_gaussian_ppo():
    """Test the Gaussian PPO agent"""
    
    # Setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = get_uav_config()
    env = UAVGenAIEnv(config)
    
    # Set seeds
    env.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")

    # Initialize agent
    agent = Gaussian_PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        lr=3e-4,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        norm_adv=True,
        ent_coef=0.01
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=10000, device=device)

    # Test sampling action
    print("Testing action sampling...")
    state = env.reset()
    action = agent.sample_action(state)
    print(f"Sampled action shape: {action.shape}, Action: {action}")
    
    # Test training
    print("Testing training...")
    
    # Collect some experience
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state

    # Train the agent
    if replay_buffer.size > 32:
        metrics = agent.train(replay_buffer, iterations=5, batch_size=32)
        print("Training metrics:", metrics)
        print("✅ Training successful!")
    else:
        print("❌ Not enough samples for training")

    # Test model saving/loading
    print("Testing model saving/loading...")
    test_dir = "test_models"
    os.makedirs(test_dir, exist_ok=True)
    
    agent.save_model(test_dir)
    print("✅ Model saved successfully!")
    
    # Test action sampling after training
    print("Testing action sampling after training...")
    state = env.reset()
    action = agent.sample_action(state)
    print(f"Sampled action after training: {action}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_gaussian_ppo() 