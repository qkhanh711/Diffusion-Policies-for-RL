import gym
from gym import spaces
import numpy as np


def EnvConfig_v1(envName):
    print(f"Using environment configuration for: {envName}")
    return {
        "num_users": 10,
        "T": 10,
        "sys_tau":0.5,
        "Gmax": 1e10,
        "Mmax": 96,  # Tăng từ 48 để không bị chạm trần memory quá sớm
        "lambda_qos": 0.75,
        "lambda_latency": 0.5,
        "lambda_mem": 1.0,
        "lambda_flops": 1.0,
        "lambda_serve_complete": 0.8,  # 80% completion threshold for feasible rewards
        "PVM": 1e12,
        "Rmem": 2.304e12,
        "max_denoise_steps": 35,
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
        # MINIMAL BONUSES: Drastically reduced to prevent overfitting
        "user_service_bonus": 3,           # Reduced from 10 to 2
        "memory_utilization_bonus": 5,     # Reduced from 30 to 5
        "fairness_bonus": 3,               # Reduced from 20 to 3
        "min_utilization_threshold": 0.55,  # Increased from 0.5 to 0.7 (very strict)
        "penalty_reduction_factor": 0.5,   # Giảm từ 0.75 để penalty nhẹ hơn, agent dễ học hơn
        # MINIMAL: Progressive bonus parameters
        "efficiency_bonus_scale": 0.1,     # Reduced from 0.5 to 0.1
        "service_quality_weight": 0.05,    # Reduced from 0.3 to 0.05
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
        self.image_size = np.random.uniform(2000, 5000)  # bytes - tăng từ [700,3000] để bài toán nặng hơn
        self.prompt_size = np.random.uniform(10, 100)   # bytes
        self.direction = np.random.uniform(0, 2*np.pi)
        self.qos_required = 30
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
        return (t_up + t_mem + t_comp + t_down) * 10, flops

    def _compute_flops(self, user, denoise_steps):
        rho = user.image_size / self.config["base_image_size"]
        return rho * (self.config["GE0"] + self.config["GD0"] + denoise_steps * self.config["G_eps"] + self.config["G_prompt"])

    def _compute_memory(self, user):
        # Assume image size = H*W, infer H*W
        return (self.config["c1"] * user.image_size + self.config["c2"]) * 1.3

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
        # print(1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm)
        return 1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm

    def _compute_reward(self, action):
        total_reward = 0
        total_penalty = 0
        total_latency = 0
        total_flops = 0
        total_mem = 0
        served = 0
        info = {}

        # Memory resets each step
        current_memory = 0
        served_users = []
        rejected_users = []
        quality_scores = []
        
        # Process all users in this step
        for i, user in enumerate(self.users):
            serve_decision = action[2 * i]
            denoise_action = action[2 * i + 1]
            
            # Convert continuous action to binary serve decision
            serve = 1 if serve_decision >= 0.5 else 0
            
            if serve:
                denoise_steps = int(np.clip(denoise_action, 1, self.config["max_denoise_steps"]))
                
                # Calculate resource requirements for this user
                mem = self._compute_memory(user)
                latency, flops = self._compute_latency(user, denoise_steps)
                
                # Check if we can serve this user (memory constraint for this step)
                if current_memory + mem <= self.config["Mmax"]:
                    # Serve this user
                    qos = self._compute_qos(denoise_steps)
                    price = self._compute_price(mem, flops, user.image_size + user.prompt_size)
                    
                    # Individual user penalties (nearly full penalty)
                    penalty_q = self.config["lambda_qos"] * self.config["penalty_reduction_factor"] * max(0, qos - user.qos_required)
                    penalty_l = self.config["lambda_latency"] * self.config["penalty_reduction_factor"] * max(0, latency - self.config["sys_tau"])

                    # DOMINANT SIGNAL: Price reward (main incentive)
                    total_reward += price
                    
                    # TINY BONUS: Minimal service bonus to encourage serving users
                    total_reward += self.config["user_service_bonus"]
                    
                    total_penalty += penalty_q + penalty_l
                    total_latency += latency
                    total_flops += flops
                    current_memory += mem
                    served += 1
                    served_users.append(i)
                    
                    # Track quality but with minimal weight
                    quality_score = max(0, user.qos_required - qos) / user.qos_required if user.qos_required > 0 else 0
                    quality_scores.append(quality_score)
                else:
                    # Cannot serve this user due to memory constraint
                    rejected_users.append(i)

        # Set total memory for this step
        total_mem = current_memory

        # System-level constraint penalties (thống nhất sử dụng sys_tau * num_users)
        if total_latency > self.config["sys_tau"] * self.config["num_users"]:
            total_penalty += self.config["lambda_latency"] * self.config["penalty_reduction_factor"] * (total_latency - self.config["sys_tau"] * self.config["num_users"])
        if total_flops > self.config["Gmax"]:
            total_penalty += self.config["lambda_flops"] * self.config["penalty_reduction_factor"] * (total_flops - self.config["Gmax"])
        if total_mem > self.config["Mmax"]:
            total_penalty += self.config["lambda_mem"] * self.config["penalty_reduction_factor"] * (total_mem - self.config["Mmax"])

        # MINIMAL BONUSES (only if very high performance):
        memory_bonus = 0
        fairness_bonus = 0
        efficiency_bonus = 0
        service_quality_bonus = 0
        
        service_ratio = served / self.config["num_users"] if self.config["num_users"] > 0 else 0
        memory_utilization = total_mem / self.config["Mmax"] if self.config["Mmax"] > 0 else 0
        
        # Only give bonuses for exceptional performance
        if service_ratio >= self.config["lambda_serve_complete"]:  # Only if serving 80%+ of users
            # 1. Minimal memory utilization bonus (very conservative)
            if memory_utilization >= self.config["min_utilization_threshold"]:
                optimal_utilization = 0.8
                if memory_utilization <= optimal_utilization:
                    memory_bonus = self.config["memory_utilization_bonus"] * (memory_utilization / optimal_utilization)
                    total_reward += memory_bonus
            
            # 2. Minimal fairness bonus (only for very high service ratios)
            if service_ratio >= self.config["lambda_serve_complete"]:  # Only for 90%+ service
                fairness_bonus = self.config["fairness_bonus"] * (service_ratio - self.config["lambda_serve_complete"]) * 10  # Bonus only for extra performance
                total_reward += fairness_bonus

            # 3. Minimal efficiency bonus
            if served > 0:
                avg_mem_per_user = total_mem / served
                max_mem_per_user = self.config["Mmax"] / self.config["num_users"]
                
                if avg_mem_per_user < max_mem_per_user * 0.5:  # Only if very efficient
                    mem_efficiency = 1 - (avg_mem_per_user / max_mem_per_user)
                    efficiency_bonus = self.config["efficiency_bonus_scale"] * mem_efficiency
                    total_reward += efficiency_bonus

            # 4. Minimal service quality bonus
            if quality_scores and len(quality_scores) >= self.config["num_users"] * 0.8:  # Only if serving most users
                avg_quality = np.mean(quality_scores)
                if avg_quality > 0.5:  # Giảm từ 0.7 xuống 0.5 để dễ đạt bonus hơn
                    service_quality_bonus = self.config["service_quality_weight"] * (avg_quality - 0.5) * 5
                    total_reward += service_quality_bonus

        # Minimal system constraint bonus (very strict conditions)
        bonus = 0
        if (total_latency <= self.config["sys_tau"] * self.config["num_users"] and 
            total_flops <= self.config["Gmax"] and 
            total_mem <= self.config["Mmax"] and
            service_ratio >= self.config["lambda_serve_complete"]):  # Only if serving 80%+ AND meeting all constraints
            bonus = self.config["psi"] * 0.2  # Very small fixed bonus

        # Store metrics
        info['served'] = served
        info['total_latency'] = total_latency
        info['total_flops'] = total_flops
        info['total_mem'] = total_mem
        info['bonus'] = bonus
        info['penalty'] = total_penalty
        info['memory_utilization'] = memory_utilization
        info['service_ratio'] = service_ratio
        info['served_users'] = served_users
        info['rejected_users'] = rejected_users
        info['fairness_bonus'] = fairness_bonus
        info['memory_bonus'] = memory_bonus
        info['efficiency_bonus'] = efficiency_bonus
        info['service_quality_bonus'] = service_quality_bonus
        info['user_service_bonus'] = served * self.config["user_service_bonus"]
        info['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0

        final_reward = total_reward - total_penalty + bonus
        return final_reward, info

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