import torch
from jlgo.uav_env import UAVGenAIEnv
import numpy as np
from tqdm import tqdm
import json
# from main import ReplayBuffer  # Nếu có ReplayBuffer riêng cho UAVEnv, import ở đây

def debug_env(num = 10):
    config = get_config()
    env = UAVGenAIEnv(config)

    print("=== ENV DEBUG START ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial state shape:", env._get_state().shape)

    # Reset env
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result
    print("\nInitial state sample:", state)

    # Test step loop
    for step in range(num):  
        action = env.action_space.sample()  # random action
        step_result = env.step(action)

        if len(step_result) == 5:  # Gym>=0.26
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Gym cũ
            next_state, reward, done, info = step_result

        print(f"\nStep {step + 1}:")
        print("  Action:", action)
        print("  Reward:", reward)
        print("  Done:", done)
        print("  Next state shape:", np.array(next_state).shape)
        if isinstance(info, dict):
            print("  Info:", info)

        state = next_state
        if done:
            print("Episode ended early.")
            break

    print("\n=== ENV DEBUG END ===")

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
        "f_inf_BS": 2.0e9,
        "Cin_BS": 0.03125,
        "lambda_Q": 0.5,
        "lambda_L": 0.5,
        "lambda_E": 1.0,
        "psi": 10.0,
        "Qreq": [np.random.randint(30, 40) for _ in range(5)],
        "T": 10
    }

def train_agent(env, agent, device, num_episodes=10000, batch_size=64, algo='ql', test = False):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)

    episode_rewards = []
    loss_log = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        losses = []

        while not done:
            action = agent.sample_action(state)
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

            if replay_buffer.size > batch_size:
                loss = agent.train(replay_buffer, iterations=1, batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)

        episode_rewards.append(total_reward)

        # Average losses per episode (if applicable)
        avg_loss = {}
        if losses:
            keys = losses[0].keys()
            for k in keys:
                avg_loss[k] = np.mean([l[k] for l in losses])
        else:
            keys = ['total_loss', 'bc_loss', 'ql_loss', 'actor_loss', 'critic_loss', 'ppo_loss', 'value_loss', 'entropy_loss']
            avg_loss = {k: 0.0 for k in keys}

        loss_log.append(avg_loss)
        if episode % (num_episodes // 10) == 0:
            # loss_log.append(avg_loss)
            print(f"Episode {episode+1}: Reward = {total_reward} \n Loss = {avg_loss}")
        # print(f"Episode {episode+1}: Reward = {total_reward}, Loss = {avg_loss}")

        # Save reward log
        np.save(f"testUAV/episode_rewards_{algo}.npy", np.array(episode_rewards))

        # Save loss log
        with open(f"testUAV/loss_log_{algo}.json", "w") as f:
            json.dump(loss_log, f, indent=4)
    if test:
        debug_env(num = 3)

    print(f"Saving reward log to testUAV/episode_rewards_{algo}.npy")
    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")

def run(algo='ql', test = False):
    # seed = 1234
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    config = get_config()
    env = UAVGenAIEnv(config)
    state_dim = env._get_state().shape[0]
    action_dim = env.num_users + 3
    max_action = 1.0  # hoặc thiết lập phù hợp nếu có action_space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algo == 'dql':
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
            lr=0.0005
        )

    elif algo == 'gppo':
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
    elif algo == 'gdql':
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
    elif algo == 'diffppo':
        from agents.diffusion_ppo import Diffusion_PPO as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=0.0008
    )
    elif algo == 'a2c':
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
        raise ValueError(f"Unknown algo: {algo}")

    print(f"Initialized agent: {agent}")
    if test:
        train_agent(env, agent, device, num_episodes=10, batch_size=64, algo=algo, test = test)
    else:
        train_agent(env, agent, device, num_episodes=2000, batch_size=64, algo=algo, test = test)

def test(test = False):
    run(algo='ppo', test = test)
    run(algo='gppo', test = test)
    run(algo='gdql', test = test)
    run(algo='a2c', test = test)
    run(algo='dql', test = test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ql")
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()
    print(f"Running {args.algo}")
    if args.test:
        test(test = args.test)
    else:
        run(args.algo, test = args.test)
    # test(test = args.test)
