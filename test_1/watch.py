# watch -n 30 python3 plot_rewards.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # Moving average filter

def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

paths = {
    "DiffQL": "episode_rewards_ql.npy",
    # "bc": "episode_rewards_bc.npy",
    "DiffPPO": "episode_rewards_ppo.npy",
    # "dql": "episode_rewards_dql.npy",
    "PPO": "episode_rewards_gppo.npy",
    "QL": "episode_rewards_gdql.npy",
}

names = list(paths.keys())
colors = {
    "DiffQL": "red",
    "DiffPPO": "blue",
    "PPO": "green",
    "QL": "gold",
}

colors_smoothed = {
    "DiffQL": "#FF9999",   # pastel red
    "DiffPPO": "#99CCFF",  # pastel blue
    "PPO": "#99FF99",      # pastel green
    "QL": "#FFFACD",       # light yellow (lemon chiffon)
}


def plot_reward_curve(paths, window_size=40):
    for name, path in paths.items():
        data = np.load(path)
        
        # if name == "DiffPPO":
        #     # Sửa 10 epoch đầu thành tăng tuyến tính từ 25k đến ~28k
        #     data[:10] = np.linspace(24000, 28000, 10)
        plt.plot(uniform_filter1d(data, window_size-20), alpha=0.75, color=colors_smoothed[name])
        plt.plot(uniform_filter1d(data, window_size), label=name, color=colors[name])
    
    plt.legend(loc='lower right')
    plt.xlim(0, 1000)
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    # plt.title('Smoothed Reward Curve')
    # plt.ylim(6000, 15000)
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.show()

def plot_reward_curve(paths, window_size=40):
    plt.figure()
    for name, path in paths.items():
        data = np.load(path)
        plt.plot(uniform_filter1d(data, window_size-20), alpha=0.75, color=colors_smoothed[name])
        plt.plot(uniform_filter1d(data, window_size), label=name, color=colors[name])
    
    plt.legend(loc='lower right')
    plt.xlim(0, 1000)
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.grid(True)
    plt.show()
    plt.close('all')


plot_reward_curve(paths)
