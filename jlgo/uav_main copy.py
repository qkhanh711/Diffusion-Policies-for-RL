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
        'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'no',
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
                      lr=0.0005)
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

    print_banner("Training Start", separator="*", num_star=90)
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        total_reward = 0
        for _ in range(args.num_episodes_per_epoch):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            done = False
            ep_reward = 0
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
                state = next_state
                ep_reward += reward
                if replay_buffer.size > args.batch_size:
                    loss = agent.train(replay_buffer, iterations=1, batch_size=args.batch_size)
            total_reward += ep_reward
        avg_reward = total_reward / args.num_episodes_per_epoch
        rewards.append(avg_reward)
        
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
    
    # Final validation
    print_banner("FINAL VALIDATION", separator="=", num_star=90)
    validate_environment(env, agent, state_dim, action_dim, device, args.num_epochs)
    
    print(f"Training completed. Results saved to {output_dir}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Final average reward: {np.mean(rewards[-10:]):.4f}")
    print(f"Best average reward: {np.max(rewards):.4f}")
    print(f"Rewards trend: {rewards[-1] - rewards[0]:.4f} (final - initial)")

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

    train_agent(env, state_dim, action_dim, max_action, device, results_dir, args)
