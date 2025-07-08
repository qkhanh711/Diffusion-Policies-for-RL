import torch
from env import GAIServiceEnv
from main import ReplayBuffer

def get_config():
    return {
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
    }

def train_agent(env, agent, device, num_episodes=10, batch_size=64):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
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
    env = GAIServiceEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
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
    run(algo='dql')
    

