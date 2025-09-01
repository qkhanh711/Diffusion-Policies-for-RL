import gym
from gym import spaces
import numpy as np


def EnvConfig_v1(envName):
    print(f"Using environment configuration for: {envName}")
    return {
        "num_users": 10,
        "T": 10,
        "sys_tau": 0.55,  # Tăng từ 0.5 lên 1.0 - cho phép latency cao hơn
        "Gmax": 1e10,
        "Mmax": 48,
        "lambda_qos": 0.1,  # Giảm từ 0.5 xuống 0.1 - penalty rất nhỏ
        "lambda_latency": 0.2,  # Giảm từ 0.8 xuống 0.2 - penalty rất nhỏ
        "lambda_mem": 0.5,  # Giảm từ 1.0 xuống 0.5
        "lambda_flops": 0.5,  # Giảm từ 1.0 xuống 0.5
        "lambda_over_provision": 0.01,  # Giảm từ 0.05 xuống 0.01 - gần như không phạt
        "lambda_excessive_steps": 0.05,  # Giảm từ 0.2 xuống 0.05 - gần như không phạt
        "PVM": 1e12,
        "Rmem": 2.304e12,
        "max_denoise_steps": 30,
        "reasonable_max_steps": 25,  # Tăng từ 25 lên 30 - cho phép nhiều steps hơn
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
        self.image_size = np.random.uniform(100, 1000)
        self.prompt_size = np.random.uniform(100, 1000)
        self.direction = np.random.uniform(0, 2*np.pi)
        # More varied QoS requirements
        self.qos_required = np.random.choice([15, 20, 25, 30, 35, 40], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        self.mobility_speed = np.random.uniform(0.5, 2.0)
        self.mobility_angle = self.direction

    def update_position(self):
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        new_position = self.position + np.array([dx, dy])
        new_position = np.clip(new_position, -1000, 1000)
        self.position = new_position
        self.mobility_angle += np.random.uniform(-0.1, 0.1)


class GAIServiceEnv_v1(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.users = [User(i, config) for i in range(config["num_users"])]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * config["num_users"],), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * config["num_users"],), dtype=np.float32)
        self.reset()

    def reset(self):
        self.time_step = 0
        for user in self.users:
            user.reset()
        return self._get_state()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        reward, info = self._compute_reward(action)
        self._move_users()
        self.time_step += 1
        done = self.time_step >= self.config["T"]
        return self._get_state(), reward, done, info

    def _get_state(self):
        state = []
        for user in self.users:
            norm_pos_x = user.position[0] / 1000.0
            norm_pos_y = user.position[1] / 1000.0
            norm_image_size = user.image_size / 10000.0
            norm_prompt_size = user.prompt_size / 1000.0
            norm_direction = user.direction / (2 * np.pi)
            norm_qos = user.qos_required / 50.0
            
            state.extend([norm_pos_x, norm_pos_y, norm_image_size, norm_prompt_size, norm_direction, norm_qos])
        return np.array(state, dtype=np.float32)

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _distance(self, user):
        user_pos_3d = np.concatenate([user.position, [0]])
        distance = np.linalg.norm(self.config["sp_pos"] - user_pos_3d)
        return max(distance, 1.0)

    def _channel_rate(self, distance):
        h_i = self.config["h0"] / (distance ** self.config["path_loss"])
        B_i = self.config["bandwidth"] / self.config["num_users"]
        snr = self.config["upload_power"] * h_i / (B_i * self.config["noise_power"])
        rate = B_i * np.log2(1 + snr)
        return max(rate / 8, 1.0)

    def _compute_latency(self, user, denoise_steps):
        d = self._distance(user)
        rate_up = self._channel_rate(d)
        rate_down = self._channel_rate(d)
        mem_rate = self.config["Rmem"] / 8
        compute_power = max(self.config["PVM"], 1.0)

        flops = self._compute_flops(user, denoise_steps)
        t_up = (user.image_size + user.prompt_size) / rate_up
        t_mem = (user.image_size + user.prompt_size) / mem_rate
        t_comp = flops / compute_power
        t_down = user.image_size / rate_down
        total_latency = (t_up + t_mem + t_comp + t_down) * 100 
        # print(total_latency) 
        return total_latency, flops

    def _compute_flops(self, user, denoise_steps):
        rho = user.image_size / self.config["base_image_size"]
        flops = rho * (self.config["GE0"] + self.config["GD0"] + denoise_steps * self.config["G_eps"] + self.config["G_prompt"])
        return max(flops, 1.0)

    def _compute_memory(self, user):
        base_memory = self.config["c1"] * user.image_size + self.config["c2"]
        return max(base_memory * 1.3, 0.1)

    def _compute_qos(self, denoise_steps):
        # More deterministic QoS mapping for better trade-off learning
        if denoise_steps < 5:
            return 45.0  # Poor quality
        elif denoise_steps < 10:
            return 35.0  # Fair quality
        elif denoise_steps < 15:
            return 25.0  # Good quality
        elif denoise_steps < 25:
            return 18.0  # Very good quality
        else:
            return 12.0  # Excellent quality

    def _qos_to_min_steps(self, required_qos):
        """Convert required QoS to minimum denoise steps needed"""
        if required_qos >= 40:
            return 1  # Very low quality requirement
        elif required_qos >= 30:
            return 5  # Low quality requirement  
        elif required_qos >= 20:
            return 10  # Medium quality requirement
        elif required_qos >= 15:
            return 15  # High quality requirement
        else:
            return 25  # Very high quality requirement

    def _compute_price_with_qos(self, mem, flops, comm, qos_achieved, qos_required):
        """Compute price with much higher base pricing to encourage serving"""
        # Tăng base price rất mạnh để khuyến khích phục vụ
        base_price = 1e-8 * mem + 1e-11 * flops + 1e-7 * comm  # Tăng gấp 10 lần nữa
        
        # QoS bonus
        qos_bonus_factor = 1.0
        if qos_achieved <= qos_required:
            qos_bonus_factor = 2.0  # Tăng từ 1.5 lên 2.0 - bonus rất lớn
        elif qos_achieved < qos_required - 10:
            qos_bonus_factor = 1.3  # Tăng từ 1.1 lên 1.3
        
        return max(base_price * qos_bonus_factor, 0.1)  # Tăng minimum price gấp 10

    def _estimate_shadow_prices(self, remaining_flops, remaining_mem, remaining_users):
        """Estimate shadow prices based on resource scarcity"""
        flops_util = 1.0 - (remaining_flops / self.config["Gmax"])
        mem_util = 1.0 - (remaining_mem / self.config["Mmax"])
        
        # Shadow prices increase exponentially as resources become scarce
        shadow_price_flops = np.exp(5 * max(0, flops_util - 0.7))  # Start penalty at 70% usage
        shadow_price_mem = np.exp(5 * max(0, mem_util - 0.7))
        
        # Opportunity cost increases with more users waiting
        opportunity_multiplier = 1.0 + 0.1 * remaining_users
        
        return shadow_price_flops * opportunity_multiplier, shadow_price_mem * opportunity_multiplier

    def _optimize_denoise_steps(self, user, base_steps, remaining_flops, remaining_mem, remaining_users):
        """Optimize denoise steps with minimal penalties to encourage serving"""
        min_steps = self._qos_to_min_steps(user.qos_required)
        max_feasible_steps = min(base_steps, self.config["max_denoise_steps"])
        
        # Start from minimum required steps
        optimal_steps = min_steps
        best_marginal_value = -np.inf
        
        # Get shadow prices - giảm mạnh
        shadow_flops, shadow_mem = self._estimate_shadow_prices(remaining_flops, remaining_mem, remaining_users)
        shadow_flops *= 0.1  # Giảm 90% shadow price
        shadow_mem *= 0.1    # Giảm 90% shadow price
        
        # Test different step counts
        for steps in range(min_steps, max_feasible_steps + 1):
            mem = self._compute_memory(user)
            latency, flops = self._compute_latency(user, steps)
            qos = self._compute_qos(steps)
            
            # Check basic feasibility - rất lỏng lẻo
            if (flops > remaining_flops * 1.2 or  # Cho phép vượt 20%
                mem > remaining_mem * 1.2 or     # Cho phép vượt 20%
                latency > self.config["sys_tau"] * 1.5):  # Cho phép vượt 50%
                break
                
            # Calculate marginal revenue
            price = self._compute_price_with_qos(mem, flops, 
                                               user.image_size + user.prompt_size, 
                                               qos, user.qos_required)
            
            # Calculate marginal cost - giảm rất mạnh
            marginal_flops_cost = shadow_flops * (flops - self._compute_flops(user, min_steps)) * 0.1
            marginal_mem_cost = shadow_mem * (mem - self._compute_memory(user)) * 0.01
            
            # Latency penalty - giảm rất mạnh
            latency_penalty = 0
            if latency > self.config["sys_tau"]:  # Chỉ phạt khi vượt limit
                latency_ratio = latency / self.config["sys_tau"]
                latency_penalty = self.config["lambda_latency"] * latency_ratio * 0.1
            
            # QoS over-provision penalty - gần như bỏ
            over_provision_penalty = 0
            if qos < user.qos_required - 20:  # Chỉ phạt khi quá tốt 20 điểm
                over_provision_penalty = self.config["lambda_over_provision"] * (user.qos_required - qos) * 0.1
            
            # Excessive steps penalty - gần như bỏ
            excessive_steps_penalty = 0
            if steps > self.config["reasonable_max_steps"] + 5:  # Cho phép vượt 5 steps
                excess = steps - (self.config["reasonable_max_steps"] + 5)
                excessive_steps_penalty = self.config["lambda_excessive_steps"] * excess * 0.1
            
            # Total marginal value
            marginal_value = (price - marginal_flops_cost - marginal_mem_cost - 
                            latency_penalty - over_provision_penalty - excessive_steps_penalty)
            
            if marginal_value > best_marginal_value:
                best_marginal_value = marginal_value
                optimal_steps = steps
                
        return optimal_steps

    def _compute_user_value_density(self, user, denoise_steps):
        """Compute value density for user prioritization with increased latency weight"""
        mem = self._compute_memory(user)
        latency, flops = self._compute_latency(user, denoise_steps)
        qos = self._compute_qos(denoise_steps)
        
        # Expected net reward
        price = self._compute_price_with_qos(mem, flops, user.image_size + user.prompt_size, qos, user.qos_required)
        qos_penalty = max(0, qos - user.qos_required) * self.config["lambda_qos"]
        
        # Add excessive steps penalty
        excessive_steps_penalty = 0
        if denoise_steps > self.config["reasonable_max_steps"]:
            excessive_steps_penalty = self.config["lambda_excessive_steps"] * (denoise_steps - self.config["reasonable_max_steps"])
        
        net_reward = price - qos_penalty - excessive_steps_penalty
        
        # Resource consumption with MUCH higher weight for latency
        resource_consumption = (flops / self.config["Gmax"] + 
                              mem / self.config["Mmax"] + 
                              latency / self.config["sys_tau"] * 5.0)  # 5x weight for latency
        
        # Value density = reward per unit resource
        return net_reward / max(resource_consumption, 0.001)

    def _compute_reward(self, action):
        total_reward = 0
        total_penalty = 0
        served = 0
        info = {}

        total_memory = 0
        total_flops = 0
        total_latency = 0
        max_individual_latency = 0
        total_excessive_steps_penalty = 0
        
        # Process user decisions with smarter step selection
        user_decisions = []
        for i, user in enumerate(self.users):
            serve_decision = action[2 * i]
            denoise_action = action[2 * i + 1]
            
            serve_prob = (serve_decision + 1.0) / 2.0
            serve = 1 if serve_prob >= 0.5 else 0
            
            if serve:
                # Raw denoise steps from action (still use agent's choice as baseline)
                denoise_normalized = (denoise_action + 1.0) / 2.0
                raw_denoise_steps = max(1, int(1 + denoise_normalized * (self.config["max_denoise_steps"] - 1)))
                
                user_decisions.append({
                    'user_idx': i,
                    'user': user,
                    'raw_denoise_steps': raw_denoise_steps,
                    'serve_prob': serve_prob
                })
        
        # Calculate remaining resources for shadow pricing
        remaining_flops = self.config["Gmax"]
        remaining_mem = self.config["Mmax"]
        remaining_users = len(user_decisions)
        
        # Optimize denoise steps for each user and calculate value density
        for decision in user_decisions:
            optimal_steps = self._optimize_denoise_steps(
                decision['user'], 
                decision['raw_denoise_steps'],
                remaining_flops,
                remaining_mem, 
                remaining_users
            )
            decision['optimal_denoise_steps'] = optimal_steps
            decision['value_density'] = self._compute_user_value_density(decision['user'], optimal_steps)
        
        # Sort by value density (highest first) instead of just distance
        user_decisions.sort(key=lambda x: x['value_density'], reverse=True)
        
        served_users = []
        rejected_users = []
        
        for decision in user_decisions:
            user = decision['user']
            denoise_steps = decision['optimal_denoise_steps']
            
            # Calculate requirements
            mem = self._compute_memory(user)
            latency, flops = self._compute_latency(user, denoise_steps)
            
            # Rất lỏng lẻo constraint checking - cho phép serve hầu hết users
            memory_ok = (total_memory + mem) <= self.config["Mmax"] * 1.5  # Cho phép vượt 50%
            flops_ok = (total_flops + flops) <= self.config["Gmax"] * 1.5   # Cho phép vượt 50%
            latency_ok = latency <= self.config["sys_tau"] * 2.0             # Cho phép vượt 100%
            
            if memory_ok and flops_ok and latency_ok:
                # Serve user
                qos = self._compute_qos(denoise_steps)
                price = self._compute_price_with_qos(mem, flops, user.image_size + user.prompt_size, qos, user.qos_required)
                
                # Giảm tất cả penalty xuống gần 0
                qos_penalty = max(0, qos - user.qos_required) * self.config["lambda_qos"] * 0.1
                
                over_provision_penalty = 0
                if qos < user.qos_required - 20:
                    over_provision_penalty = self.config["lambda_over_provision"] * (user.qos_required - qos) * 0.1
                
                individual_latency_penalty = 0
                if latency > self.config["sys_tau"]:
                    latency_ratio = latency / self.config["sys_tau"]
                    individual_latency_penalty = self.config["lambda_latency"] * latency_ratio * 0.05  # Rất nhỏ
                
                excessive_steps_penalty = 0
                if denoise_steps > self.config["reasonable_max_steps"] + 5:
                    excess = denoise_steps - (self.config["reasonable_max_steps"] + 5)
                    excessive_steps_penalty = self.config["lambda_excessive_steps"] * excess * 0.05
                
                total_reward += price
                total_penalty += (qos_penalty + over_provision_penalty + 
                                individual_latency_penalty + excessive_steps_penalty)
                total_excessive_steps_penalty += excessive_steps_penalty
                
                total_memory += mem
                total_flops += flops
                total_latency += latency
                max_individual_latency = max(max_individual_latency, latency)
                served += 1
                
                # Update remaining resources
                remaining_flops -= flops
                remaining_mem -= mem
                remaining_users -= 1
                
                served_users.append({
                    'user_idx': decision['user_idx'],
                    'denoise_steps': denoise_steps,
                    'optimal_steps': denoise_steps,
                    'raw_steps': decision['raw_denoise_steps'],
                    'served': True,
                    'qos': qos,
                    'qos_required': user.qos_required,
                    'latency': latency,
                    'value_density': decision['value_density'],
                    'individual_latency_penalty': individual_latency_penalty,
                    'excessive_steps_penalty': excessive_steps_penalty,
                    'distance': self._distance(user)
                })
            else:
                # Penalty rất nhỏ cho rejection
                rejected_users.append({
                    'user_idx': decision['user_idx'],
                    'reason': 'resource_constraint',
                    'served': False,
                    'requested_steps': denoise_steps,
                    'latency_violation': not latency_ok,
                    'memory_violation': not memory_ok,
                    'flops_violation': not flops_ok
                })
                # Penalty rất nhỏ
                total_penalty += 0.001 * self.config["psi"]

        # System constraint penalties - giảm rất mạnh hoặc bỏ
        if total_memory > self.config["Mmax"] * 2.0:  # Chỉ phạt khi vượt quá 200%
            total_penalty += self.config["lambda_mem"] * (total_memory - self.config["Mmax"] * 2.0) * 0.1
        if total_flops > self.config["Gmax"] * 2.0:   # Chỉ phạt khi vượt quá 200%
            total_penalty += self.config["lambda_flops"] * (total_flops - self.config["Gmax"] * 2.0) * 0.1

        # Bỏ system latency penalty
        total_latency_limit = self.config["sys_tau"] * self.config["num_users"] * 3.0  # Rất lỏng
        # Không phạt system latency nữa

        # Tăng rất mạnh bonus system
        bonus = 0
        # Luôn có bonus cơ bản
        bonus += self.config["psi"] * 0.5  # Tăng từ 0.3 lên 0.5
            
        # Service ratio bonus - tăng rất mạnh
        service_ratio = served / self.config["num_users"] if self.config["num_users"] > 0 else 0
        service_bonus = service_ratio * self.config["psi"] * 2.0  # Tăng từ 1.0 lên 2.0
        
        # Efficiency bonus - dễ đạt hơn
        efficiency_bonus = 0
        if self.config["Mmax"] > 0 and self.config["Gmax"] > 0:
            mem_util = min(total_memory / self.config["Mmax"], 2.0)  # Cho phép 200%
            flops_util = min(total_flops / self.config["Gmax"], 2.0)  # Cho phép 200%
            
            # Bonus cho việc sử dụng tài nguyên
            efficiency_bonus = (mem_util + flops_util) * self.config["psi"] * 0.2  # Tăng từ 0.1

        # Smart allocation bonus
        optimization_bonus = 0
        if served_users:
            steps_saved = sum(max(0, user['raw_steps'] - user['optimal_steps']) for user in served_users)
            optimization_bonus = steps_saved * 0.3  # Tăng từ 0.2

        # Low latency bonus - không cần thiết nữa
        low_latency_bonus = 0

        # Serving bonus - tăng rất mạnh
        serving_bonus = served * self.config["psi"] * 0.3  # Tăng từ 0.1 lên 0.3

        # Bonus cho việc serve nhiều user
        if served >= 7:  # Bonus lớn nếu serve ít nhất 7/10 users
            high_service_bonus = self.config["psi"] * 1.0
        elif served >= 5:  # Bonus trung bình nếu serve ít nhất 5/10 users
            high_service_bonus = self.config["psi"] * 0.5
        else:
            high_service_bonus = 0

        info['served'] = served
        info['rejected'] = len(rejected_users)
        info['total_mem'] = total_memory
        info['total_flops'] = total_flops
        info['total_latency'] = total_latency
        info['max_latency'] = max_individual_latency
        info['avg_latency'] = total_latency / served if served > 0 else 0
        info['latency_limit'] = total_latency_limit
        info['bonus'] = bonus + service_bonus + efficiency_bonus + optimization_bonus + serving_bonus + high_service_bonus
        info['penalty'] = total_penalty
        info['service_ratio'] = service_ratio
        info['optimization_bonus'] = optimization_bonus
        info['serving_bonus'] = serving_bonus
        info['high_service_bonus'] = high_service_bonus
        info['total_excessive_steps_penalty'] = total_excessive_steps_penalty
        info['resource_utilization'] = {
            'memory': total_memory / self.config["Mmax"] if self.config["Mmax"] > 0 else 0,
            'flops': total_flops / self.config["Gmax"] if self.config["Gmax"] > 0 else 0,
            'latency': total_latency / total_latency_limit if total_latency_limit > 0 else 0
        }
        info['served_users'] = served_users
        info['rejected_users'] = rejected_users

        final_reward = (total_reward - total_penalty + bonus + service_bonus + 
                       efficiency_bonus + optimization_bonus + serving_bonus + high_service_bonus)
        return final_reward, info

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    state = env.reset()
    print("Initial State shape:", state.shape)
    print("State range:", state.min(), "to", state.max())
    
    # Test với nhiều random actions
    total_served = 0
    total_tests = 10
    
    for test in range(total_tests):
        state = env.reset()  # Reset environment for each test
        action = np.random.uniform(-1, 1, size=2 * config["num_users"])
        next_state, reward, done, info = env.step(action)
        total_served += info['served']
        
        print(f"\n=== Test {test + 1} ===")
        print(f"Reward: {reward:.4f}")
        print(f"Served users: {info['served']}/{config['num_users']}")
        print(f"Service ratio: {info['service_ratio']:.2%}")
        print(f"Base reward: {reward - info['bonus']:.2f}")
        print(f"Total bonus: {info['bonus']:.2f}")
        print(f"  - Service bonus: {info['service_ratio'] * config['psi'] * 2.0:.2f}")
        print(f"  - Serving bonus: {info['serving_bonus']:.2f}")
        print(f"  - High service bonus: {info.get('high_service_bonus', 0):.2f}")
        print(f"Total penalty: {info['penalty']:.2f}")
        
        if info['served_users']:
            avg_steps = np.mean([u['optimal_steps'] for u in info['served_users']])
            print(f"Average denoise steps: {avg_steps:.1f}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total served across {total_tests} tests: {total_served}")
    print(f"Average served per test: {total_served / total_tests:.1f}")
    print(f"Expected serve rate: {(total_served / total_tests) / config['num_users']:.1%}")