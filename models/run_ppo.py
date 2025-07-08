import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import GAIServiceEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import numpy as np

def get_default_config():
    return {
        "num_users": 10,
        "T": 100,
        "tau": 5.0,
        "Gmax": 1e12,
        "Mmax": 8e9,
        "lambda_qos": 10.0,
        "lambda_latency": 5.0,
        "lambda_mem": 1.0,
        "lambda_flops": 1.0,
        "max_denoise_steps": 50,
    }

if __name__ == "__main__":
    config = get_default_config()
    env = GAIServiceEnv(config)

    # Kiểm tra tính hợp lệ của env
    check_env(env, warn=True)

    # Chọn thuật toán RL
    algo = "PPO"  # Đổi thành "A2C" nếu muốn
    if algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
        reward_file = "episode_rewards_ppo.npy"
    elif algo == "A2C":
        model = A2C("MlpPolicy", env, verbose=1)
        reward_file = "episode_rewards_a2c.npy"
    else:
        raise ValueError("Chỉ hỗ trợ PPO hoặc A2C")

    # Train
    model.learn(total_timesteps=100_000)

    # Đánh giá và lưu reward từng episode
    episode_rewards = []
    n_episodes = 100
    obs, *_ = env.reset()
    for i in range(n_episodes):
        done = False
        obs, *_ = env.reset()
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            episode_reward += reward
        episode_rewards.append(episode_reward)
    np.save(reward_file, np.array(episode_rewards))
    print(f"Saved episode rewards to {reward_file}")
    print(f"Total reward over {n_episodes} episodes: {np.sum(episode_rewards)}")
