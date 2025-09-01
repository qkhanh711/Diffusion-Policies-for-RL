import gymnasium as gym
import numpy as np

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Gộp các Box trong Dict thành 1 Box lớn
        self.action_keys = list(env.action_space.spaces.keys())
        self.action_dims = [int(np.prod(env.action_space.spaces[k].shape)) for k in self.action_keys]
        low = np.concatenate([env.action_space.spaces[k].low.flatten() for k in self.action_keys])
        high = np.concatenate([env.action_space.spaces[k].high.flatten() for k in self.action_keys])
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        # Chuyển action vector thành Dict cho env gốc
        action_dict = {}
        idx = 0
        for k, d in zip(self.action_keys, self.action_dims):
            action_dict[k] = action[idx:idx+d]
            idx += d
        return action_dict