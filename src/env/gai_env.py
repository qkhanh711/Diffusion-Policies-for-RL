import gym
from gym import spaces
import numpy as np

def EnvConfig(envName):
    """Configuration for the environment"""
    print(f"Using environment configuration for: {envName}")
    return {
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


class User:
    def __init__(self, user_id, config):
        self.user_id = user_id
        self.reset(config)
        self.config = config

    def reset(self, config=None):
        if config is None:
            config = self.config
        self.position = np.random.uniform(0, 100, size=2)
        self.image_size = np.random.uniform(100, 1000)
        self.prompt_size = np.random.uniform(10, 100)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.qos_required = 30 
        self.mobility_speed = np.random.uniform(0.5, 2.0)  # tốc độ di chuyển
        self.mobility_angle = self.direction

    def update_position(self):
        # Công thức (10-11): cập nhật vị trí dựa trên hướng và tốc độ
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        self.position += np.array([dx, dy])
        # Thay đổi hướng ngẫu nhiên nhẹ
        self.mobility_angle += np.random.uniform(-0.1, 0.1)

class GAIServiceEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        N = config["num_users"]
        # Trạng thái người dùng (vector hóa)
        self.pos = np.zeros((N, 2), dtype=np.float32)
        self.image_size = np.zeros(N, dtype=np.float32)
        self.prompt_size = np.zeros(N, dtype=np.float32)
        self.direction = np.zeros(N, dtype=np.float32)
        self.qos_required = np.zeros(N, dtype=np.float32)
        self.mobility_speed = np.zeros(N, dtype=np.float32)
        self.mobility_angle = np.zeros(N, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 * N,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.tile([0, 1], N),
            high=np.tile([1, config["max_denoise_steps"]], N),
            dtype=np.float32
        )
        self.reset()

    # ---------------------- Vectorized state ----------------------
    def reset(self):
        cfg = self.config
        N = cfg["num_users"]
        self.time_step = 0

        self.pos = np.random.uniform(0, 100, size=(N, 2)).astype(np.float32)
        self.image_size = np.random.uniform(100, 1000, size=N).astype(np.float32)
        self.prompt_size = np.random.uniform(10, 100, size=N).astype(np.float32)
        self.direction = np.random.uniform(0, 2*np.pi, size=N).astype(np.float32)
        self.qos_required = np.random.uniform(10, 50, size=N).astype(np.float32)
        self.mobility_speed = np.random.uniform(0.5, 2.0, size=N).astype(np.float32)
        self.mobility_angle = self.direction.copy()
        return self._get_state()

    def step(self, action):
        reward, info = self._compute_reward_vectorized(action)
        self._move_users_vectorized()
        self.time_step += 1
        done = self.time_step >= self.config["T"]
        return self._get_state(), reward, done, info

    def _get_state(self):
        # [x, y, image_size, prompt_size, direction, qos_required] * N
        return np.concatenate([
            self.pos[:, 0],
            self.pos[:, 1],
            self.image_size,
            self.prompt_size,
            self.direction,
            self.qos_required
        ], axis=0).astype(np.float32)

    def _move_users_vectorized(self):
        self.pos[:, 0] += self.mobility_speed * np.cos(self.mobility_angle)
        self.pos[:, 1] += self.mobility_speed * np.sin(self.mobility_angle)
        self.mobility_angle += np.random.uniform(-0.1, 0.1, size=self.pos.shape[0]).astype(np.float32)

    # ---------------------- Vectorized helpers ----------------------
    def _distance_all(self):
        # UAV at sp_pos (3D), users at z=0
        sp = self.config["sp_pos"].astype(np.float32)  # [x,y,z]
        # users 3D = [x,y,0]
        users3d = np.concatenate([self.pos, np.zeros((self.pos.shape[0], 1), dtype=np.float32)], axis=1)
        d = np.linalg.norm(sp[None, :] - users3d, axis=1)  # [N]
        return d + 1e-9  # tránh chia 0

    def _channel_rate_all(self, distance):
        cfg = self.config
        h_i = cfg["h0"] / np.power(distance, cfg["path_loss"])
        B_i = cfg["bandwidth"] / cfg["num_users"]
        snr = cfg["upload_power"] * h_i / (B_i * cfg["noise_power"])
        rate = B_i * np.log2(1.0 + snr)
        return (rate / 8.0).astype(np.float64)  # bytes/s, dùng float64 cho ổn định

    def _compute_flops_all(self, steps):
        cfg = self.config
        rho = self.image_size / cfg["base_image_size"]
        # GE0 + GD0 + steps*G_eps + G_prompt
        per = (cfg["GE0"] + cfg["GD0"] + steps * cfg["G_eps"] + cfg["G_prompt"])
        return (rho * per).astype(np.float64)

    def _compute_memory_all(self):
        cfg = self.config
        return (cfg["c1"] * self.image_size + cfg["c2"]).astype(np.float64)

    def _compute_qos_all(self, steps):
        base_fid = 50.0
        noise = np.random.normal(0.0, 2.0, size=steps.shape[0])
        return (base_fid / (1.0 + 0.1 * steps) + noise).astype(np.float64)

    def _compute_price_all(self, mem, flops, comm):
        # 1e-7*mem + 1e-5*flops + 2.5e-6*comm
        return (1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm).astype(np.float64)

    # ---------------------- Vectorized reward ----------------------
    def _compute_reward_vectorized(self, action):
        cfg = self.config
        N = cfg["num_users"]

        # Parse action -> serves in {0,1}, steps in [1, max]
        serves = (action[0::2] >= 0.5).astype(np.float64)                        # [N]
        steps = np.clip(action[1::2].astype(np.int32), 1, cfg["max_denoise_steps"]).astype(np.float64)

        # Comm rates (upload & download assumed same model)
        d = self._distance_all()                                                 # [N]
        rate_up = self._channel_rate_all(d)                                      # [N] bytes/s
        rate_down = rate_up                                                      # cùng kênh

        # Compute per-user quantities
        mem = self._compute_memory_all()                                         # [N]
        flops = self._compute_flops_all(steps)                                   # [N]
        qos = self._compute_qos_all(steps)                                       # [N]
        comm = (self.image_size + self.prompt_size).astype(np.float64)           # [N]
        price = self._compute_price_all(mem, flops, comm)                        # [N]

        # Latency components
        mem_rate = cfg["Rmem"] / 8.0  # bytes/s
        compute_power = cfg["PVM"]    # FLOPs/s

        t_up = comm / np.maximum(rate_up, 1e-9)
        t_mem = comm / np.maximum(mem_rate, 1e-9)
        t_comp = flops / np.maximum(compute_power, 1e-9)
        t_down = self.image_size.astype(np.float64) / np.maximum(rate_down, 1e-9)
        latency = t_up + t_mem + t_comp + t_down                                 # [N]

        # Apply serve mask
        mask = serves
        price_s   = price   * mask
        latency_s = latency * mask
        flops_s   = flops   * mask
        mem_s     = mem     * mask
        qos_s     = qos     * mask
        qos_req_s = self.qos_required.astype(np.float64) * mask

        # Reward & penalties
        relu = np.maximum
        total_reward = price_s.sum()

        # per-user penalties
        pen_q = cfg["lambda_qos"]     * relu(qos_s - qos_req_s, 0.0)
        pen_l = cfg["lambda_latency"] * relu(latency_s - cfg["sys_tau"], 0.0)
        total_penalty = (pen_q + pen_l).sum()

        # system totals
        total_latency = latency_s.sum()
        total_flops   = flops_s.sum()
        total_mem     = mem_s.sum()

        # system penalties
        total_penalty += (
            cfg["lambda_latency"] * relu(total_latency - cfg["sys_tau"] * N, 0.0) +
            cfg["lambda_flops"]   * relu(total_flops   - cfg["Gmax"],        0.0) +
            cfg["lambda_mem"]     * relu(total_mem     - cfg["Mmax"],        0.0)
        )

        # bonus
        bonus = cfg["psi"] if (
            (total_latency <= cfg["sys_tau"] * N) and
            (total_flops   <= cfg["Gmax"])      and
            (total_mem     <= cfg["Mmax"])
        ) else 0.0

        info = dict(
            served=int(mask.sum()),
            total_latency=float(total_latency),
            total_flops=float(total_flops),
            total_mem=float(total_mem),
            bonus=float(bonus),
            penalty=float(total_penalty)
        )
        return float(total_reward - total_penalty + bonus), info

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    config = EnvConfig("GAIServiceEnv")
    env = GAIServiceEnv(config)
    obs = env.reset()
    print("Initial Observation:", obs)
    
    action = np.random.uniform(0, 1, size=(2 * env.num_users,))
    action[1::2] = np.random.randint(1, env.max_denoise_steps + 1, size=env.num_users)  # denoise steps
    print("Sample Action:", action)
    
    next_obs, reward, done, info = env.step(action)
    print("Next Observation:", next_obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    