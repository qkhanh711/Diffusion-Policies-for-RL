import torch
from uav_env import UAVGenAIEnv
import numpy as np
# from main import ReplayBuffer  # Nếu có ReplayBuffer riêng cho UAVEnv, import ở đây

# Dummy ReplayBuffer nếu chưa có
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


def get_config():
    return {
        "num_users": 5,
        "Dmax": 20,
        "tau": 3.0,
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
        "T": 10
    }

def train_agent(env, agent, device, num_episodes=10, batch_size=64):
    state_dim = env._get_state().shape[0]
    action_dim = env.num_users + 3
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=10000, device=device)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

            if replay_buffer.size > batch_size:
                agent.train(replay_buffer, iterations=1, batch_size=batch_size)
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward}")

    print(f"Average reward over {num_episodes} episodes: {sum(episode_rewards)/num_episodes}")

def run(algo='ql'):
    config = get_config()
    env = UAVGenAIEnv(config)
    state_dim = env._get_state().shape[0]
    action_dim = env.num_users + 3
    max_action = 1.0  # hoặc thiết lập phù hợp nếu có action_space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0.99,
            tau=0.005,
            max_q_backup=1.0,
            beta_schedule='linear',
            n_timesteps=5,
            eta=0.001,
            lr=0.0003,
            lr_decay=0.99,
            lr_maxt=1000,
            grad_norm=1.0
        )
    elif algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0.99,
            tau=0.005,
            beta_schedule='linear',
            n_timesteps=5,
            lr=0.0003
        )
    elif algo == 'ppo':
        from agents.ppo_diffusion import Diffusion_PPO as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0.99,
            tau=0.005,
            beta_schedule='linear',
            n_timesteps=5,
            lr=0.0003,
            clip_ratio=0.2,
            value_clip_ratio=0.2,
            norm_adv=True
        )
    elif algo == 'dql':
        from agents.dql_diffusion import Diffusion_DQL as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"Initialized agent: {agent}")
    train_agent(env, agent, device, num_episodes=10, batch_size=64)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ql")
    args = parser.parse_args()
    print(f"Running {args.algo}")
    run(args.algo)