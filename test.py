import torch
from env import GAIServiceEnv
from main import ReplayBuffer
import numpy as np
import csv
from tqdm import tqdm
import json

def get_config():
    return {
        "num_users": 3,
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
        "tau": 3.0,
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


def train_agent(env, agent, device, num_episodes=10000, batch_size=64, algo='ql'):
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
        np.save(f"test/episode_rewards_{algo}.npy", np.array(episode_rewards))

    # Save loss log
        with open(f"test/loss_log_{algo}.json", "w") as f:
            json.dump(loss_log, f, indent=4)

    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")

def run(algo='ql'):
    # seed = 1234
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    config = get_config()
    env = GAIServiceEnv(config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
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
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim, 
                      max_action=max_action,
                      device=device,
                      lr=0.0005)
    # elif algo == 'dql':
    #     from agents.dql_diffusion import Diffusion_DQL as Agent
    #     agent = Agent(
    #         state_dim=state_dim,
    #         action_dim=action_dim,
    #         max_action=max_action,
    # #         device=device
    #     )
    elif algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            lr=0.0005,                # giảm learning rate
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
    train_agent(env, agent, device, num_episodes=1000, batch_size=128, algo=algo)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ql")
    args = parser.parse_args()
    print(f"Running {args.algo}")
    run(args.algo)