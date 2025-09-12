import gym
from gym import spaces
import numpy as np


def EnvConfig_v1(envName):
    print(f"Using environment configuration for: {envName}")
    return {
        "num_users": 10,
        "T": 10,
        "sys_tau": 0.1,
        "Gmax": 5e7,
        "Mmax": 48,
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
        "psi": 100,
        "qos_required": 30,
    }


class User:
    def __init__(self, user_id, config):
        self.user_id = user_id
        self.config = config
        self.reset(config)

    def reset(self, config=None):
        if config is None:
            config = self.config
        self.position = np.random.uniform(-500, 500, size=2)
        self.image_size = np.random.uniform(100, 1000)  # bytes
        self.prompt_size = np.random.uniform(10, 100)   # bytes
        self.direction = np.random.uniform(0, 2*np.pi)
        self.qos_required = config["qos_required"]
        self.mobility_speed = np.random.uniform(0.5, 2.0)
        self.mobility_angle = self.direction

    def update_position(self):
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        self.position += np.array([dx, dy])
        self.mobility_angle += np.random.uniform(-0.1, 0.1)


class GAIServiceEnv_v1(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.users = [User(i, config) for i in range(config["num_users"])]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * config["num_users"],), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, 1] * config["num_users"]),
                                       high=np.array([1, config["max_denoise_steps"]] * config["num_users"]),
                                       dtype=np.float32)
        self.reset()

    def reset(self):
        self.time_step = 0
        for user in self.users:
            user.reset()
        return self._get_state()

    def step(self, action):
        reward, info = self._compute_reward(action)
        self._move_users()
        self.time_step += 1
        done = self.time_step >= self.config["T"]
        return self._get_state(), reward, done, info

    def _get_state(self):
        state = []
        for user in self.users:
            state.extend([user.position[0], user.position[1], user.image_size, user.prompt_size, user.direction, user.qos_required])
        return np.array(state, dtype=np.float32)

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _distance(self, user):
        user_pos_3d = np.concatenate([user.position, [0]])
        return np.linalg.norm(self.config["sp_pos"] - user_pos_3d)

    def _channel_rate(self, distance):
        h_i = self.config["h0"] / (distance ** self.config["path_loss"])
        B_i = self.config["bandwidth"] / self.config["num_users"]
        snr = self.config["upload_power"] * h_i / (B_i * self.config["noise_power"])
        rate = B_i * np.log2(1 + snr)
        return rate / 8  # bit → byte

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
        if denoise_steps < 5:
            return np.random.uniform(40, 50)  # Poor quality (high BRISQUE)
        elif denoise_steps < 10:
            return np.random.uniform(30, 40)  # Fair quality
        elif denoise_steps < 15:
            return np.random.uniform(20, 30)  # Good quality
        else:
            return np.random.uniform(10, 20)  # Excellent quality (low BRISQUE)

    def _compute_price(self, mem, flops, comm):
        return 1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm
    
    def _served_penalty(self, serve):
        serve_ratio = serve / self.config["num_users"]
        if serve_ratio < 0.3:
            return 20
        elif serve_ratio < 0.6:
            return 10
        elif serve_ratio < 0.8:
            return 5
        else:
            return -10

    def _compute_reward(self, action):
        total_reward = 0
        total_penalty = 0
        total_latency = 0
        total_flops = 0
        total_mem = 0
        info = {}
        total_serve = 0
        qos_gained = []
        denoise_steps_list = []
        for i, user in enumerate(self.users):

            # serve = int(round(action[2 * i]))
            # denoise_steps = int(round(action[2 * i + 1]))
            # denoise_steps = np.clip(denoise_steps, 1, self.config["max_denoise_steps"])

 
            serve = float(action[2*i])
            denoise_steps = float(action[2*i + 1])
            normalized_serve = (serve + 1.0) / 2.0 if serve < 0 or serve > 1 else serve
            normalized_denoise = (denoise_steps + 1.0) / 2.0 if denoise_steps < 0 or denoise_steps > 1 else denoise_steps
            serve = 1 if normalized_serve >= 0.5 else 0
            normalized_denoise = np.clip(normalized_denoise, 0.0, 1.0)
            scaled_denoise_steps = int(1 + normalized_denoise * (self.config["max_denoise_steps"] - 1))
            denoise_steps = np.clip(scaled_denoise_steps, 1, self.config["max_denoise_steps"])

            
            
            denoise_steps_list.append(denoise_steps)
            # print(f"DEBUG: User {i}, Raw Action: Serve={serve}, Denoise Steps={denoise_steps}, Normalized Serve={normalized_serve}, Normalized Denoise={normalized_denoise}, Scaled Denoise Steps={scaled_denoise_steps}")                       
            # print(f"Action for User {i}: Serve={serve}, Denoise Steps={denoise_steps}")
            if serve:
                # print(f"Serving {serve} User {i} with Denoise Steps: {denoise_steps}")
                latency, flops = self._compute_latency(user, denoise_steps)
                # print(f"Computed Latency for User {i}: {latency}")
                mem = self._compute_memory(user)
                qos = self._compute_qos(denoise_steps)
                qos_gained.append(qos)
                price = self._compute_price(mem, flops, user.image_size + user.prompt_size)
                penalty_q = self.config["lambda_qos"] * max(0, qos - user.qos_required)
                penalty_l = self.config["lambda_latency"] * max(0, latency - self.config["sys_tau"])

                total_reward += price
                total_penalty += penalty_q + penalty_l
                total_latency += latency
                total_flops += flops
                total_mem += mem
                total_serve += serve
            mean_qos = np.mean(qos_gained) if qos_gained else None
            # print(f"User {i}: Serve={serve}, Denoise Steps={denoise_steps}, Latency={latency if serve else 'N/A'}, Flops={flops if serve else 'N/A'}, Mem={mem if serve else 'N/A'}, QoS={qos if serve else 'N/A'}, Price={price if serve else 'N/A'}, Penalty QoS={penalty_q if serve else 'N/A'}, Penalty Latency={penalty_l if serve else 'N/A'}")
            info.update({
                "serve": serve,
                "total_served": total_serve,
                "total_latency": total_latency,
                "total_flops": total_flops,
                "total_mem": total_mem,
                "bonus": 0,  # Placeholder, will be updated later
                "penalty": total_penalty,
                "penalty_qos": penalty_q if serve else 0,
                "penalty_latency": penalty_l if serve else 0,
                "mean_qos": mean_qos,
                "latency": latency if serve else None,
                "flops": flops if serve else None,
                "mem": mem if serve else None,
                "price": price if serve else None,
                "denoise_steps": denoise_steps_list if serve else 0,
            })

        if total_latency > self.config["sys_tau"]:
            total_penalty += self.config["lambda_latency"] * (total_latency - self.config["sys_tau"])
        if total_flops > self.config["Gmax"]:
            total_penalty += self.config["lambda_flops"] * (total_flops - self.config["Gmax"])
        if total_mem > self.config["Mmax"]:
            total_penalty += self.config["lambda_mem"] * (total_mem - self.config["Mmax"])
        total_penalty += self._served_penalty(total_serve) 
        bonus = 0
        if total_latency <= self.config["sys_tau"] and total_flops <= self.config["Gmax"] and total_mem <= self.config["Mmax"]:
            bonus = self.config["psi"]
        info["bonus"] = bonus

        return total_reward - total_penalty + bonus, info

    # def _compute_reward(self, action):
    #     cfg = self.config
    #     N = cfg["num_users"]

    #     total_reward = 0.0
    #     total_penalty = 0.0
    #     total_latency = 0.0
    #     total_flops = 0.0
    #     total_mem = 0.0
    #     served = 0
    #     info = {}

    #     relu = lambda x: x if x > 0 else 0

    #     for i, user in enumerate(self.users):
    #         # serve ∈ {0,1}
    #         serve = 1 if (action[2*i] >= 0.5) else 0

    #         # clip số bước denoise
    #         denoise_steps = int(action[2*i + 1])
    #         if denoise_steps < 1:
    #             denoise_steps = 1
    #         elif denoise_steps > cfg["max_denoise_steps"]:
    #             denoise_steps = cfg["max_denoise_steps"]

    #         if serve:
    #             latency, flops = self._compute_latency(user, denoise_steps)
    #             mem = self._compute_memory(user)
    #             qos = self._compute_qos(denoise_steps)
    #             price = self._compute_price(mem, flops, user.image_size + user.prompt_size)

    #             total_reward += price
    #             total_penalty += (
    #                 cfg["lambda_qos"]     * relu(qos     - user.qos_required) +
    #                 cfg["lambda_latency"] * relu(latency - cfg["sys_tau"])
    #             )
    #             total_latency += latency
    #             total_flops   += flops
    #             total_mem     += mem
    #             served        += 1

    #     # Phạt ràng buộc hệ thống
    #     total_penalty += (
    #         cfg["lambda_latency"] * relu(total_latency - cfg["sys_tau"] * N) +
    #         cfg["lambda_flops"]   * relu(total_flops   - cfg["Gmax"]) +
    #         cfg["lambda_mem"]     * relu(total_mem     - cfg["Mmax"])
    #     )

    #     # Bonus nếu thỏa cả 3 ràng buộc hệ thống
    #     ok_latency = (total_latency <= cfg["sys_tau"] * N)
    #     ok_flops   = (total_flops   <= cfg["Gmax"])
    #     ok_mem     = (total_mem     <= cfg["Mmax"])
    #     bonus = cfg["psi"] if (ok_latency and ok_flops and ok_mem) else 0.0

    #     info.update(dict(
    #         served=served,
    #         total_latency=total_latency,
    #         total_flops=total_flops,
    #         total_mem=total_mem,
    #         bonus=bonus,
    #         penalty=total_penalty
    #     ))
    #     return total_reward - total_penalty + bonus, info


if __name__ == "__main__":
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    state = env.reset()
    print("Initial State:", state)
    action = np.random.uniform(0, 1, size=2 * config["num_users"])
    next_state, reward, done, info = env.step(action)
    print("Next State:", next_state, "Reward:", reward, "Done:", done, "Info:", info)