import gym
from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
        self.qos_required = np.random.uniform(10, 50)  # QoS yêu cầu (ví dụ FID)
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
        self.num_users = config["num_users"]
        self.T = 5
        self.tau = 2.0
        self.Gmax = 1e12
        self.Mmax = 8e9
        self.lambda_qos = 10.0
        self.lambda_latency = 5.0
        self.lambda_mem = 1.0
        self.lambda_flops = 1.0
        self.time_step = 0
        self.users = [User(i, config) for i in range(self.num_users)]
        self.max_denoise_steps = config.get("max_denoise_steps", 50)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6*self.num_users,), dtype=np.float32)
        # Action: [b1, D1, b2, D2, ..., bN, DN]
        act_low = np.array([0, 1] * self.num_users)
        act_high = np.array([1, self.max_denoise_steps] * self.num_users)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.reset()

    def reset(self):
        self.time_step = 0
        for user in self.users:
            user.reset()
        obs = self._get_state()
        return obs

    # def reset(self, seed=None, options=None):
    #     self.time_step = 0
    #     for user in self.users:
    #         user.reset()
    #     obs = self._get_state()
    #     info = {}
    #     return obs, info

    def step(self, action):
        reward, info = self._compute_reward(action)
        self._move_users()
        self.time_step += 1
        done = self.time_step >= self.T
        obs = self._get_state()
        return obs, reward, done, info

    # def step(self, action):
    #     reward, info = self._compute_reward(action)
    #     self._move_users()
    #     self.time_step += 1
    #     done = self.time_step >= self.T
    #     truncated = False
    #     obs = self._get_state()
    #     return obs, reward, done, truncated, info
    
    def _get_state(self):
        state = []
        for user in self.users:
            state.extend([
                user.position[0], user.position[1],
                user.image_size, user.prompt_size,
                user.direction, user.qos_required
            ])
        return np.array(state, dtype=np.float32)

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _compute_latency(self, user, denoise_steps):
        # Công thức mô phỏng latency (có thể điều chỉnh theo paper)
        upload_rate = 10e6  # bytes/s
        mem_rate = 10e6     # bytes/s
        compute_power = 1e9 # FLOPs/s
        download_rate = 10e6 # bytes/s
        output_size = user.image_size  # giả sử output size = image size
        flops = denoise_steps * 1e8  # mỗi bước denoise cần 1e8 FLOPs
        tup = (user.image_size + user.prompt_size) / upload_rate
        tmem = (user.image_size + user.prompt_size) / mem_rate
        tcomp = flops / compute_power
        tdown = output_size / download_rate
        total_latency = tup + tmem + tcomp + tdown
        return total_latency, flops

    def _compute_qos(self, denoise_steps):
        # FID giảm khi tăng số bước denoise (giả lập)
        base_fid = 50
        fid = base_fid / (1 + 0.1 * denoise_steps)
        return fid

    def _compute_price(self, user, flops):
        # Công thức pricing (14)
        price = 0.1 * user.image_size + 0.05 * user.prompt_size + 0.00001 * flops
        return price

    def _compute_reward(self, action):
        total_revenue = 0
        total_penalty = 0
        total_latency = 0
        total_flops = 0
        total_mem = 0
        info = {"user_rewards": [], "user_penalties": [], "user_latencies": [], "user_qos": []}
        for i, user in enumerate(self.users):
            serve = int(round(action[2*i]))
            denoise_steps = int(round(action[2*i+1]))
            denoise_steps = max(1, min(denoise_steps, self.max_denoise_steps))
            if serve == 1:
                latency, flops = self._compute_latency(user, denoise_steps)
                qos = self._compute_qos(denoise_steps)
                price = self._compute_price(user, flops)
                mem = user.image_size + user.prompt_size  # memory usage
                penalty_qos = self.lambda_qos * max(0, qos - user.qos_required)
                penalty_latency = self.lambda_latency * max(0, latency - self.tau)
                total_revenue += price
                total_penalty += penalty_qos + penalty_latency
                total_latency += latency
                total_flops += flops
                total_mem += mem
                info["user_rewards"].append(price)
                info["user_penalties"].append(penalty_qos + penalty_latency)
                info["user_latencies"].append(latency)
                info["user_qos"].append(qos)
            else:
                info["user_rewards"].append(0)
                info["user_penalties"].append(0)
                info["user_latencies"].append(0)
                info["user_qos"].append(0)
        # Penalty nếu tổng latency, flops, memory vượt ngưỡng
        if total_latency > self.tau * self.num_users:
            total_penalty += self.lambda_latency * (total_latency - self.tau * self.num_users)
        if total_flops > self.Gmax:
            total_penalty += self.lambda_flops * (total_flops - self.Gmax)
        if total_mem > self.Mmax:
            total_penalty += self.lambda_mem * (total_mem - self.Mmax)
        # Thưởng nếu thỏa tất cả ràng buộc
        bonus = 0
        if total_latency <= self.tau * self.num_users and total_flops <= self.Gmax and total_mem <= self.Mmax:
            bonus = 1
        reward = total_revenue - total_penalty + bonus
        info["total_revenue"] = total_revenue
        info["total_penalty"] = total_penalty
        info["total_latency"] = total_latency
        info["total_flops"] = total_flops
        info["total_mem"] = total_mem
        info["bonus"] = bonus
        return reward, info
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]



# Example config for GAIServiceEnv
# config = {
#     "num_users": 10,
#     "max_time": 100,
#     "latency_limit": 5.0,      # seconds
#     "max_flops": 1e12,         # FLOPs
#     "max_vram": 8e9,           # VRAM bytes
#     "penalty_qos": 10.0,
#     "penalty_latency": 5.0,
# }