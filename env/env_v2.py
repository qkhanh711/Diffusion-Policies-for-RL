import gym
from gym import spaces
import numpy as np


def EnvConfig_v1(envName):
    print(f"Using environment configuration for: {envName}")
    config = {
        "num_users": 10,
        "T": 10,
        "sys_tau": 3,
        "Gmax": 1e13,
        "Mmax": 128e9,
        "lambda_qos": 0.5,
        "lambda_latency": 0.5,
        "lambda_mem": 1.0,
        "lambda_flops": 1.0,
        "PVM": 1e12,
        "Rmem": 2.304e12,
        "max_denoise_steps": 50,
        "c1": 3.81e-6,
        "c2": 4.86,
        "base_image_size": 1024 * 1024,
        "GE0": 1e8,
        "GD0": 1e8,
        "G_eps": 1e8,
        "G_prompt": 1e7,
        "sp_pos": np.array([0.0, 0.0, 50.0]),
        "h0": 1.42e-4,
        "path_loss": 2.0,
        "bandwidth": 1e6,
        "noise_power": 4.0e-21,
        "upload_power": 0.0501,
        "download_power": 0.5012,
        "psi": 10,
    }
    
    # Add normalization constants for faster training
    config.update({
        "state_norm": {
            "position": 100.0,
            "image_size": 1000.0,
            "prompt_size": 100.0,
            "direction": 2 * np.pi,
            "qos": 50.0
        },
        "reward_scale": 1e-6,  # Scale down rewards
        "use_cache": True  # Enable caching
    })
    return config


class User:
    def __init__(self, user_id, config):
        self.user_id = user_id
        self.config = config
        # Pre-allocate arrays for better performance
        self.position = np.zeros(2)
        self.reset(config)

    def reset(self, config=None):
        if config is None:
            config = self.config
        # Use fixed seed for reproducibility and faster debugging
        np.random.seed(self.user_id)
        self.position[:] = np.random.uniform(0, 100, size=2)
        self.image_size = np.random.uniform(100, 1000)
        self.prompt_size = np.random.uniform(10, 100)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.qos_required = np.random.uniform(10, 50)
        self.mobility_speed = np.random.uniform(0.5, 2.0)
        self.mobility_angle = self.direction

    def update_position(self):
        # Vectorized position update
        cos_angle = np.cos(self.mobility_angle)
        sin_angle = np.sin(self.mobility_angle)
        self.position[0] += self.mobility_speed * cos_angle
        self.position[1] += self.mobility_speed * sin_angle
        self.mobility_angle += np.random.uniform(-0.1, 0.1)


class GAIServiceEnv_v1(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.users = [User(i, config) for i in range(config["num_users"])]
        
        # Normalized observation space
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(6 * config["num_users"],), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, 1] * config["num_users"]),
            high=np.array([1, config["max_denoise_steps"]] * config["num_users"]),
            dtype=np.float32
        )
        
        # Pre-compute constants for performance
        self._precompute_constants()
        
        # Cache for expensive computations
        self._distance_cache = {}
        self._rate_cache = {}
        
        self.reset()

    def _precompute_constants(self):
        """Pre-compute constants to avoid repeated calculations"""
        self.B_i = self.config["bandwidth"] / self.config["num_users"]
        self.mem_rate = self.config["Rmem"] / 8
        self.noise_B = self.B_i * self.config["noise_power"]
        
        # Pre-compute normalization factors
        self.state_norms = np.array([
            self.config["state_norm"]["position"],
            self.config["state_norm"]["position"],
            self.config["state_norm"]["image_size"],
            self.config["state_norm"]["prompt_size"],
            self.config["state_norm"]["direction"],
            self.config["state_norm"]["qos"]
        ] * self.config["num_users"])

    def reset(self):
        self.time_step = 0
        for user in self.users:
            user.reset()
        # Clear caches
        self._distance_cache.clear()
        self._rate_cache.clear()
        return self._get_state()

    def step(self, action):
        reward, info = self._compute_reward(action)
        self._move_users()
        self.time_step += 1
        done = self.time_step >= self.config["T"]
        # Clear position-dependent caches after movement
        self._distance_cache.clear()
        self._rate_cache.clear()
        return self._get_state(), reward, done, info

    def _get_state(self):
        """Optimized state generation with normalization"""
        state = np.zeros(6 * self.config["num_users"], dtype=np.float32)
        
        for i, user in enumerate(self.users):
            base_idx = i * 6
            state[base_idx:base_idx+6] = [
                user.position[0], user.position[1], 
                user.image_size, user.prompt_size, 
                user.direction, user.qos_required
            ]
        
        # Normalize state
        return state / self.state_norms

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _distance(self, user):
        """Cached distance computation"""
        if self.config.get("use_cache", False) and user.user_id in self._distance_cache:
            return self._distance_cache[user.user_id]
            
        user_pos_3d = np.concatenate([user.position, [0]])
        distance = np.linalg.norm(self.config["sp_pos"] - user_pos_3d)
        
        if self.config.get("use_cache", False):
            self._distance_cache[user.user_id] = distance
        return distance

    def _channel_rate(self, distance):
        """Cached channel rate computation"""
        # Round distance for cache key
        dist_key = round(distance, 2)
        if self.config.get("use_cache", False) and dist_key in self._rate_cache:
            return self._rate_cache[dist_key]
            
        h_i = self.config["h0"] / (distance ** self.config["path_loss"])
        snr = self.config["upload_power"] * h_i / self.noise_B
        rate = self.B_i * np.log2(1 + snr) / 8
        
        if self.config.get("use_cache", False):
            self._rate_cache[dist_key] = rate
        return rate

    def _compute_latency(self, user, denoise_steps):
        d = self._distance(user)
        rate_up = self._channel_rate(d)
        rate_down = self._channel_rate(d)
        mem_rate = self.config["Rmem"] / 8
        compute_power = self.config["PVM"]

        flops = self._compute_flops(user, denoise_steps)
        t_up = (user.image_size + user.prompt_size) / rate_up
        t_mem = (user.image_size + user.prompt_size) / mem_rate
        t_comp = flops / compute_power
        t_down = user.image_size / rate_down
        return t_up + t_mem + t_comp + t_down, flops

    def _compute_flops(self, user, denoise_steps):
        rho = user.image_size / self.config["base_image_size"]
        return rho * (self.config["GE0"] + self.config["GD0"] + denoise_steps * self.config["G_eps"] + self.config["G_prompt"])

    def _compute_memory(self, user):
        # Assume image size = H*W, infer H*W
        return self.config["c1"] * user.image_size + self.config["c2"]

    def _compute_qos(self, denoise_steps):
        base_fid = 50
        noise = np.random.normal(0, 2)
        return base_fid / (1 + 0.1 * denoise_steps) + noise

    def _compute_price(self, mem, flops, comm):
        return 1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm

    def _compute_reward(self, action):
        """Optimized reward computation with vectorization where possible"""
        total_reward = 0
        total_penalty = 0
        total_latency = 0
        total_flops = 0
        total_mem = 0
        info = {}

        # Vectorize action processing
        serves = np.round(action[::2]).astype(int)
        denoise_steps = np.clip(np.round(action[1::2]).astype(int), 1, self.config["max_denoise_steps"])

        for i, user in enumerate(self.users):
            if serves[i]:
                latency, flops = self._compute_latency(user, denoise_steps[i])
                mem = self._compute_memory(user)
                qos = self._compute_qos(denoise_steps[i])
                price = self._compute_price(mem, flops, user.image_size + user.prompt_size)
                
                penalty_q = self.config["lambda_qos"] * max(0, qos - user.qos_required)
                penalty_l = self.config["lambda_latency"] * max(0, latency - self.config["sys_tau"])

                total_reward += price
                total_penalty += penalty_q + penalty_l
                total_latency += latency
                total_flops += flops
                total_mem += mem

        # Vectorized constraint penalties
        constraint_violations = [
            (total_latency, self.config["sys_tau"] * self.config["num_users"], self.config["lambda_latency"]),
            (total_flops, self.config["Gmax"], self.config["lambda_flops"]),
            (total_mem, self.config["Mmax"], self.config["lambda_mem"])
        ]
        
        for value, limit, penalty_weight in constraint_violations:
            if value > limit:
                total_penalty += penalty_weight * (value - limit)

        # Bonus calculation
        bonus = 0
        if (total_latency <= self.config["sys_tau"] * self.config["num_users"] and 
            total_flops <= self.config["Gmax"] and total_mem <= self.config["Mmax"]):
            bonus = self.config["psi"]

        # Scale reward for better training stability
        final_reward = (total_reward - total_penalty + bonus) * self.config.get("reward_scale", 1.0)
        
        return final_reward, info

if __name__ == "__main__":
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    state = env.reset()
    print("Initial State:", state)
    action = np.random.uniform(0, 1, size=2 * config["num_users"])
    next_state, reward, done, info = env.step(action)
    print("Next State:", next_state, "Reward:", reward, "Done:", done, "Info:", info)