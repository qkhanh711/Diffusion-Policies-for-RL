import torch
from env import GAIServiceEnv
from main import ReplayBuffer
import numpy as np
import csv
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

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


def train_agent(env, agent, device, num_episodes=10000, batch_size=64, algo='ql', seed=1234):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=100000, device=device)

    episode_rewards = []
    loss_log = []
    
    # PPO-specific parameters
    if algo == 'ppo':
        warmup_episodes = 200       # Đủ warmup
        update_frequency = 4        # Update ít thường xuyên hơn
        ppo_epochs = 2             # Multiple epochs per update
        batch_size = 256           # Larger batch cho PPO
    else:
        warmup_episodes = 50
        update_frequency = 1  
        ppo_epochs = 1

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

            # PPO-specific training logic
            if algo == 'ppo':
                # Wait for more data and less frequent updates
                if replay_buffer.size > batch_size * 4 and episode > warmup_episodes:
                    if episode % update_frequency == 0:
                        # Multiple epochs for PPO
                        for epoch in range(ppo_epochs):
                            loss = agent.train(replay_buffer, iterations=1, batch_size=batch_size)
                            if loss is not None and all(len(v) > 0 for v in loss.values()):
                                valid_loss = {}
                                for k, v in loss.items():
                                    if all(isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val) and abs(val) < 1e10 for val in v):
                                        valid_loss[k] = v
                                    else:
                                        valid_loss[k] = [0.0]
                                
                                if valid_loss:
                                    losses.append(valid_loss)
            else:
                # Original logic for other algorithms
                if replay_buffer.size > batch_size:
                    loss = agent.train(replay_buffer, iterations=1, batch_size=batch_size)
                    if loss is not None and all(len(v) > 0 for v in loss.values()):
                        valid_loss = {}
                        for k, v in loss.items():
                            if all(isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val) and abs(val) < 1e10 for val in v):
                                valid_loss[k] = v
                            else:
                                valid_loss[k] = [0.0]
                        
                        if valid_loss:
                            losses.append(valid_loss)

        episode_rewards.append(total_reward)

        # Existing loss processing code...
        avg_loss = {}
        if losses:
            all_keys = set()
            for loss_dict in losses:
                all_keys.update(loss_dict.keys())
            
            for k in all_keys:
                try:
                    values = []
                    for loss_dict in losses:
                        if k in loss_dict and loss_dict[k]:
                            values.extend(loss_dict[k])
                    
                    if values:
                        filtered_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v) and abs(v) < 1e10]
                        if filtered_values:
                            avg_loss[k] = float(np.mean(filtered_values))
                        else:
                            avg_loss[k] = 0.0
                    else:
                        avg_loss[k] = 0.0
                except Exception as e:
                    print(f"Error processing loss {k}: {e}")
                    avg_loss[k] = 0.0
        else:
            keys = ['total_loss', 'bc_loss', 'ql_loss', 'actor_loss', 'critic_loss', 'ppo_loss', 'value_loss', 'entropy_loss']
            avg_loss = {k: 0.0 for k in keys}

        loss_log.append(avg_loss)
        
        if episode % (num_episodes // 10) == 0:
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
            print(f"Loss = {avg_loss}")
            # Debug info for PPO
            if algo == 'ppo' and hasattr(agent, 'step'):
                print(f"PPO Step: {agent.step}, Warmup: {episode > warmup_episodes}")

        # Save periodically
        if episode % 1 == 0:
            import os
            root_path = "~/khanhnq/DQL-Offline-RL"
            root_path = os.path.expanduser(root_path)
            if not os.path.exists(root_path + f"/test{seed}"):
                os.makedirs(root_path + f"/test{seed}")
            if episode % 100 == 0 or episode == num_episodes - 1:
                print(f"Saving episode rewards and loss log to {root_path}/test{seed}")
            np.save(root_path + f"/test{seed}/episode_rewards_{algo}.npy", np.array(episode_rewards))
            with open(root_path + f"/test{seed}/loss_log_{algo}.json", "w") as f:
                json.dump(loss_log, f, indent=5)
            
        # Early stopping for PPO
        if algo == 'ppo' and len(episode_rewards) > 600:
            recent_avg = np.mean(episode_rewards[-50:])
            if recent_avg < 10000:  # Target reward cho convergence
                print(f"PPO converged at episode {episode} with avg reward {recent_avg}")
                break
    # Save final results
    from scipy.ndimage import uniform_filter1d  # Moving average filter
    final_rewards = np.array(episode_rewards)
    plt.plot(uniform_filter1d(final_rewards, size=50))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward Curve - {algo}')
    plt.savefig(f"/home/khanhnq/khanhnq/DQL-Offline-RL/test{seed}/reward_curve_{algo}.png")
    plt.close()
    print(f"Image saved to ~/khanhnq/khanhnq/DQL-Offline-RL/test{seed}/reward_curve_{algo}.png")


    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")

def run(algo='ql', num_episodes=4000, seed=None):
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
                      lr=3e-5,
                      clip_param=0.02,
                      entropy_coef=0.0,
                      grad_norm=0.05,
                      value_loss_coef=0.5,
                      n_timesteps=2,
                      warmup_steps=3000,
                      )     # Much smaller grad norm
    elif algo == 'gppo':
        from agents.gaussian_ppo import Gaussian_PPO as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim, 
                      max_action=max_action,
                      device=device,
                      )      # Max log std
    elif algo == 'gdql':
        from agents.gaussian_dql import Gaussian_DQL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      lr=0.0003,
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
    elif algo == 'da2c':
        from agents.a2c_diffusion import Diffusion_A2C as Agent
        agent = Agent(state_dim=state_dim,
                  action_dim=action_dim, 
                  max_action=max_action,
                  device=device,
                  lr=5e-5,              # Very small learning rate
                  gamma=0.99,
                  n_timesteps=10,       # Fewer diffusion steps
                  entropy_coef=0.02,    # Higher entropy for exploration
                  value_loss_coef=0.1,  # Lower value loss weight
                  grad_norm=0.1)        # Very small gradient clipping
    elif algo == 'a2c':
        from agents.a2c_agent import A2C_Agent as Agent
        agent = Agent(state_dim=state_dim,
                  action_dim=action_dim, 
                  max_action=max_action,
                  device=device,
                  lr=1e-4,              # Standard A2C learning rate
                  gamma=0.99,
                  entropy_coef=0.01,
                  value_loss_coef=0.5,
                  grad_norm=0.5)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"Initialized agent: {agent}")
    train_agent(env, agent, device, num_episodes=num_episodes, batch_size=128, algo=algo, seed=seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dql")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=4000)
    args = parser.parse_args()
    random_seed = np.random.randint(0, 10000)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Running {args.algo}")
        run(args.algo, num_episodes=args.num_episodes, seed=args.seed)
    else:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Using random seed: {random_seed}")
        print("No seed provided, using default random seed.")
        run(args.algo, num_episodes=args.num_episodes, seed=random_seed)
        
        # load and save reward to png
        # import matplotlib.pyplot as plt
        # rewards = np.load(f"test/test{random_seed}/episode_rewards_{args.algo}.npy")
        # plt.plot(rewards)
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.savefig(f"test/test{random_seed}/rewards_{args.algo}.png")
        # plt.close()
    print("Training complete.")

    