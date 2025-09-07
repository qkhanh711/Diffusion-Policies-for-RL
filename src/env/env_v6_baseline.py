import gym
from gym import spaces
import numpy as np
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional


class MetricsLogger:
    """
    Class để ghi lại và phân tích metrics của GAI Service Environment
    Hỗ trợ tracking real-time metrics và xuất báo cáo định kỳ
    """
    
    def __init__(self, window_size: int = 100, log_file: Optional[str] = None):
        """
        Args:
            window_size: Kích thước cửa sổ cho moving average
            log_file: Đường dẫn file để ghi log (optional)
        """
        self.window_size = window_size
        self.log_file = log_file
        
        # Episode-level metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_served_counts = deque(maxlen=window_size)
        self.episode_service_ratios = deque(maxlen=window_size)
        
        # Step-level metrics (per timestep)
        self.step_metrics = {
            'rewards': deque(maxlen=window_size * 10),
            'served_users': deque(maxlen=window_size * 10),
            'time_spent': deque(maxlen=window_size * 10),
            'peak_memory': deque(maxlen=window_size * 10),
            'total_flops': deque(maxlen=window_size * 10),
            'num_batches': deque(maxlen=window_size * 10),
            'peak_users_per_batch': deque(maxlen=window_size * 10),
            'memory_utilization': deque(maxlen=window_size * 10),
            'penalties': deque(maxlen=window_size * 10),
            'bonuses': deque(maxlen=window_size * 10),
            'prices': deque(maxlen=window_size * 10),
            # QoS metrics
            'avg_qos_achieved': deque(maxlen=window_size * 10),
            'qos_satisfaction_rate': deque(maxlen=window_size * 10),
            'avg_qos_required': deque(maxlen=window_size * 10),
            # Diffusion steps metrics
            'avg_diffusion_steps': deque(maxlen=window_size * 10),
            'min_diffusion_steps': deque(maxlen=window_size * 10),
            'max_diffusion_steps': deque(maxlen=window_size * 10),
        }
        
        # Constraint violation tracking
        self.violations = {
            'memory_violations': 0,
            'time_violations': 0,
            'flops_violations': 0,
            'qos_violations': 0,
        }
        
        # Cumulative metrics
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = time.time()
        
        # Detailed logs for analysis
        self.detailed_logs = []
        
        print(f"MetricsLogger initialized with window_size={window_size}")
        if log_file:
            print(f"Logging to file: {log_file}")
    
    def log_step(self, step_info: Dict[str, Any], reward: float, timestep: int):
        """Ghi lại metrics cho mỗi step"""
        self.total_steps += 1
        
        # Log basic metrics
        self.step_metrics['rewards'].append(reward)
        self.step_metrics['served_users'].append(step_info.get('served', 0))
        self.step_metrics['time_spent'].append(step_info.get('time_spent', 0))
        self.step_metrics['peak_memory'].append(step_info.get('peak_mem', 0))
        self.step_metrics['total_flops'].append(step_info.get('total_flops', 0))
        self.step_metrics['num_batches'].append(step_info.get('num_batches', 0))
        self.step_metrics['peak_users_per_batch'].append(step_info.get('peak_users', 0))
        self.step_metrics['memory_utilization'].append(step_info.get('memory_utilization', 0))
        self.step_metrics['penalties'].append(step_info.get('penalty', 0))
        self.step_metrics['bonuses'].append(step_info.get('bonus', 0))
        self.step_metrics['prices'].append(step_info.get('price', 0))
        
        # Log QoS metrics
        self.step_metrics['avg_qos_achieved'].append(step_info.get('avg_qos_achieved', 0))
        self.step_metrics['qos_satisfaction_rate'].append(step_info.get('qos_satisfaction_rate', 0))
        self.step_metrics['avg_qos_required'].append(step_info.get('avg_qos_required', 0))
        
        # Log diffusion steps metrics
        self.step_metrics['avg_diffusion_steps'].append(step_info.get('avg_diffusion_steps', 0))
        self.step_metrics['min_diffusion_steps'].append(step_info.get('min_diffusion_steps', 0))
        self.step_metrics['max_diffusion_steps'].append(step_info.get('max_diffusion_steps', 0))
        
        # Track constraint violations
        if step_info.get('peak_mem', 0) > step_info.get('memory_limit', float('inf')):
            self.violations['memory_violations'] += 1
        
        if step_info.get('time_spent', 0) > step_info.get('time_budget', float('inf')):
            self.violations['time_violations'] += 1
            
        if step_info.get('total_flops', 0) > step_info.get('flops_limit', float('inf')):
            self.violations['flops_violations'] += 1
        
        # Track QoS violations
        if step_info.get('qos_violations', 0) > 0:
            self.violations['qos_violations'] += step_info.get('qos_violations', 0)
        
        # Detailed log entry
        detailed_entry = {
            'episode': self.total_episodes,
            'timestep': timestep,
            'total_step': self.total_steps,
            'reward': reward,
            'timestamp': time.time() - self.start_time,
            **step_info
        }
        self.detailed_logs.append(detailed_entry)
        
        # Keep only recent detailed logs to prevent memory overflow
        if len(self.detailed_logs) > self.window_size * 20:
            self.detailed_logs = self.detailed_logs[-self.window_size * 10:]
    
    def log_episode(self, episode_reward: float, episode_length: int, total_served: int, num_users: int):
        """Ghi lại metrics cho mỗi episode"""
        self.total_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_served_counts.append(total_served)
        
        service_ratio = total_served / (num_users * episode_length) if (num_users * episode_length) > 0 else 0
        self.episode_service_ratios.append(service_ratio)
        
        # Log to file if specified
        if self.log_file:
            self._write_episode_to_file(episode_reward, episode_length, total_served, service_ratio)
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """Lấy thống kê episode gần đây"""
        if not self.episode_rewards:
            return {}
        
        return {
            'avg_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'avg_service_ratio': np.mean(self.episode_service_ratios),
            'avg_served_per_episode': np.mean(self.episode_served_counts),
            'total_episodes': self.total_episodes,
        }
    
    def get_step_statistics(self) -> Dict[str, Dict[str, float]]:
        """Lấy thống kê step-level metrics"""
        stats = {}
        
        for metric_name, values in self.step_metrics.items():
            if len(values) > 0:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'recent_avg': np.mean(list(values)[-min(10, len(values)):])  # Avg of last 10 steps
                }
        
        return stats
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê về vi phạm constraints"""
        total_steps = max(self.total_steps, 1)
        return {
            'memory_violation_rate': self.violations['memory_violations'] / total_steps,
            'time_violation_rate': self.violations['time_violations'] / total_steps,
            'flops_violation_rate': self.violations['flops_violations'] / total_steps,
            'qos_violation_rate': self.violations['qos_violations'] / total_steps,
            'total_violations': sum(self.violations.values()),
            'violation_breakdown': dict(self.violations),
        }
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Lấy xu hướng performance theo thời gian"""
        recent_episodes = min(20, len(self.episode_rewards))
        if recent_episodes < 2:
            return {}
        
        recent_rewards = list(self.episode_rewards)[-recent_episodes:]
        recent_service_ratios = list(self.episode_service_ratios)[-recent_episodes:]
        
        return {
            'reward_trend': recent_rewards,
            'service_ratio_trend': recent_service_ratios,
            'reward_moving_avg': self._moving_average(recent_rewards, min(5, recent_episodes)),
            'service_ratio_moving_avg': self._moving_average(recent_service_ratios, min(5, recent_episodes)),
        }
    
    def print_summary(self, detailed: bool = False):
        """In tóm tắt metrics"""
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        
        # Episode statistics
        episode_stats = self.get_episode_statistics()
        if episode_stats:
            print(f"\n📊 EPISODE STATISTICS (Last {len(self.episode_rewards)} episodes):")
            print(f"  Average Reward: {episode_stats['avg_episode_reward']:.4f} ± {episode_stats['std_episode_reward']:.4f}")
            print(f"  Max/Min Reward: {episode_stats['max_episode_reward']:.4f} / {episode_stats['min_episode_reward']:.4f}")
            print(f"  Average Service Ratio: {episode_stats['avg_service_ratio']:.3f}")
            print(f"  Average Episode Length: {episode_stats['avg_episode_length']:.1f}")
            print(f"  Total Episodes: {episode_stats['total_episodes']}")
        
        # Violation statistics
        violation_stats = self.get_violation_statistics()
        print(f"\n⚠️  CONSTRAINT VIOLATIONS:")
        print(f"  Memory Violations: {violation_stats['violation_breakdown']['memory_violations']} ({violation_stats['memory_violation_rate']:.3%})")
        print(f"  Time Violations: {violation_stats['violation_breakdown']['time_violations']} ({violation_stats['time_violation_rate']:.3%})")
        print(f"  FLOPs Violations: {violation_stats['violation_breakdown']['flops_violations']} ({violation_stats['flops_violation_rate']:.3%})")
        print(f"  QoS Violations: {violation_stats['violation_breakdown']['qos_violations']} ({violation_stats['qos_violation_rate']:.3%})")
        
        # QoS Statistics
        if len(self.step_metrics['avg_qos_achieved']) > 0:
            print(f"\n🎯 QoS PERFORMANCE:")
            avg_qos_achieved = np.mean(self.step_metrics['avg_qos_achieved'])
            avg_qos_required = np.mean(self.step_metrics['avg_qos_required']) if len(self.step_metrics['avg_qos_required']) > 0 else 0
            qos_satisfaction = np.mean(self.step_metrics['qos_satisfaction_rate']) if len(self.step_metrics['qos_satisfaction_rate']) > 0 else 0
            
            print(f"  Average QoS Achieved: {avg_qos_achieved:.2f}")
            print(f"  Average QoS Required: {avg_qos_required:.2f}")
            print(f"  QoS Satisfaction Rate: {qos_satisfaction:.3%}")
            print(f"  QoS Gap: {avg_qos_achieved - avg_qos_required:+.2f}")
        
        # Diffusion Steps Statistics  
        if len(self.step_metrics['avg_diffusion_steps']) > 0:
            print(f"\n🔄 DIFFUSION STEPS ANALYSIS:")
            avg_steps = np.mean(self.step_metrics['avg_diffusion_steps'])
            min_steps = np.mean(self.step_metrics['min_diffusion_steps']) if len(self.step_metrics['min_diffusion_steps']) > 0 else 0
            max_steps = np.mean(self.step_metrics['max_diffusion_steps']) if len(self.step_metrics['max_diffusion_steps']) > 0 else 0
            
            print(f"  Average Diffusion Steps: {avg_steps:.1f}")
            print(f"  Min/Max Steps: {min_steps:.1f} / {max_steps:.1f}")
            print(f"  Steps Range: {max_steps - min_steps:.1f}")
        
        # Recent performance
        if detailed:
            step_stats = self.get_step_statistics()
            print(f"\n📈 RECENT PERFORMANCE (Last {min(len(self.step_metrics['rewards']), 100)} steps):")
            for metric, stats in step_stats.items():
                if metric in ['avg_qos_achieved', 'qos_satisfaction_rate', 'avg_qos_required', 
                             'avg_diffusion_steps', 'min_diffusion_steps', 'max_diffusion_steps']:
                    print(f"  {metric}: {stats['recent_avg']:.4f} (avg: {stats['mean']:.4f})")
                elif metric in ['rewards', 'served_users', 'time_spent', 'peak_memory', 'total_flops', 
                               'num_batches', 'peak_users_per_batch', 'memory_utilization', 'penalties', 'bonuses', 'prices']:
                    print(f"  {metric}: {stats['recent_avg']:.4f} (avg: {stats['mean']:.4f})")
        
        # Runtime info
        runtime = time.time() - self.start_time
        print(f"\n⏱️  RUNTIME INFO:")
        print(f"  Total Runtime: {runtime:.1f}s")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Steps/Second: {self.total_steps/runtime:.2f}")
        
        print("="*60)
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Tính moving average"""
        if len(data) < window:
            return data
        return [np.mean(data[i:i+window]) for i in range(len(data)-window+1)]
    
    def _write_episode_to_file(self, reward: float, length: int, served: int, service_ratio: float):
        """Ghi episode data vào file"""
        try:
            with open(self.log_file, 'a') as f:
                log_entry = {
                    'episode': self.total_episodes,
                    'timestamp': time.time(),
                    'reward': reward,
                    'length': length,
                    'served': served,
                    'service_ratio': service_ratio,
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def export_detailed_logs(self, filename: str):
        """Xuất detailed logs ra file JSON"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            logs_for_export = []
            for entry in self.detailed_logs:
                converted_entry = {}
                for key, value in entry.items():
                    if isinstance(value, np.integer):
                        converted_entry[key] = int(value)
                    elif isinstance(value, np.floating):
                        converted_entry[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        converted_entry[key] = value.tolist()
                    elif isinstance(value, list):
                        # Convert any numpy types in lists
                        converted_list = []
                        for item in value:
                            if isinstance(item, (np.integer, np.floating)):
                                converted_list.append(float(item) if isinstance(item, np.floating) else int(item))
                            else:
                                converted_list.append(item)
                        converted_entry[key] = converted_list
                    else:
                        converted_entry[key] = value
                logs_for_export.append(converted_entry)
                        
            with open(filename, 'w') as f:
                json.dump(logs_for_export, f, indent=2)
            print(f"Detailed logs exported to {filename}")
        except Exception as e:
            print(f"Error exporting logs: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_metrics(self):
        """Reset tất cả metrics"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_served_counts.clear()
        self.episode_service_ratios.clear()
        
        for metric_values in self.step_metrics.values():
            metric_values.clear()
        
        self.violations = {k: 0 for k in self.violations.keys()}
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.detailed_logs.clear()
        
        print("Metrics reset successfully")


def EnvConfig_v1_baseline(envName):
    """Baseline configuration với parallel batch processing"""
    print(f"Using BASELINE environment configuration for: {envName}")
    return {
        "num_users": 10,
        "T": 1,
        "sys_tau": 3.0,  # Thay đổi: Ngân sách thời gian cho 1 phiên (session)
        "Gmax": 1e10,
        "Mmax": 48,  # Peak memory per batch
        "lambda_qos": 0.3,
        "lambda_latency": 0.3,
        "lambda_mem": 0.5,
        "lambda_flops": 0.5,
        "lambda_serve_complete": 0.8,
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
        "psi": 50,
        "user_service_bonus": 1,
        "penalty_reduction_factor": 0.3,
        # Thêm cấu hình cho parallel batching
        "session_is_batched": True,
        "session_time_budget": "sys_tau",  # Dùng sys_tau làm ngân sách thời gian phiên
        "memory_constraint": "peak_per_batch",  # Peak memory mỗi batch thay vì cộng dồn
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
        self.image_size = np.random.uniform(1500, 3000) * 8 # Tăng kích thước ảnh
        self.prompt_size = np.random.uniform(10, 100)
        self.direction = np.random.uniform(0, 2*np.pi)
        self.qos_required = 30
        self.mobility_speed = np.random.uniform(0.5, 2.0)
        self.mobility_angle = self.direction

    def update_position(self):
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        self.position += np.array([dx, dy])
        self.mobility_angle += np.random.uniform(-0.1, 0.1)


class GAIServiceEnv_v1_baseline(gym.Env):
    def __init__(self, config, enable_metrics=True, metrics_window_size=100, log_file=None):
        super().__init__()
        self.config = config
        self.users = [User(i, config) for i in range(config["num_users"])]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * config["num_users"],), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, 1] * config["num_users"]),
                                       high=np.array([1, config["max_denoise_steps"]] * config["num_users"]),
                                       dtype=np.float32)
        
        # Initialize metrics logger
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            self.metrics_logger = MetricsLogger(
                window_size=metrics_window_size,
                log_file=log_file
            )
        else:
            self.metrics_logger = None
        
        # Episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_served = 0
        
        self.reset()

    def reset(self):
        # Log previous episode if metrics enabled
        if self.enable_metrics and self.metrics_logger and hasattr(self, 'time_step'):
            if hasattr(self, 'current_episode_reward'):
                self.metrics_logger.log_episode(
                    episode_reward=self.current_episode_reward,
                    episode_length=self.time_step,
                    total_served=self.current_episode_served,
                    num_users=self.config["num_users"]
                )
        
        # Reset environment
        self.time_step = 0
        self.current_episode_reward = 0.0
        self.current_episode_served = 0
        
        for user in self.users:
            user.reset()
        return self._get_state()

    def step(self, action):
        # Sử dụng parallel batch processing
        reward, info = self._compute_reward_parallel(action)
        
        # Update episode tracking
        self.current_episode_reward += reward
        self.current_episode_served += info.get('served', 0)
        
        # Log step metrics
        if self.enable_metrics and self.metrics_logger:
            # Add constraint limits to info for violation tracking
            enhanced_info = info.copy()
            enhanced_info.update({
                'memory_limit': self.config["Mmax"],
                'time_budget': self.config["sys_tau"],
                'flops_limit': self.config["Gmax"],
            })
            self.metrics_logger.log_step(enhanced_info, reward, self.time_step)
        
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
        return rate / 8

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
        return (t_up + t_mem + t_comp + t_down), flops

    def _compute_flops(self, user, denoise_steps):
        rho = user.image_size / self.config["base_image_size"]
        return rho * (self.config["GE0"] + self.config["GD0"] + denoise_steps * self.config["G_eps"] + self.config["G_prompt"])

    def _compute_memory(self, user):
        return (self.config["c1"] * user.image_size + self.config["c2"]) * 1.5

    def _compute_qos(self, denoise_steps):
        if denoise_steps < 5:
            return np.random.uniform(35, 42)
        elif denoise_steps < 10:
            return np.random.uniform(30, 37)
        elif denoise_steps < 15:
            return np.random.uniform(25, 32)
        else:
            return np.random.uniform(13, 24)

    def _compute_price(self, mem, flops, comm):
        return 1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm

    def _estimate_price(self, mem, flops, comm):
        """Tách riêng để dùng nhiều nơi"""
        return 1e-7 * mem + 1e-5 * flops + 2.5e-6 * comm

    def _compute_reward_parallel(self, action):
        """
        Parallel session with batch scheduling:
        - Within one step, we have time budget sys_tau (e.g., 3.0s).
        - Build batches: each batch runs concurrently; time cost = max latency in that batch.
        - Per-batch memory = sum mem_i <= Mmax. We schedule batches sequentially until time budget runs out.
        - Reward = sum price of served users + small service bonus per served user - light penalties (+ optional ψ).
        """
        cfg = self.config
        time_budget = cfg["sys_tau"]     # e.g., 3.0s (ngân sách thời gian cho 1 phiên)
        Mmax       = cfg["Mmax"]

        # 1) Thu thập ứng viên (serve=1) + ước lượng metrics cho từng user
        candidates = []
        for i, user in enumerate(self.users):
            serve_decision = action[2*i]
            denoise_action = action[2*i + 1]
            if serve_decision < 0.5:
                continue
            # Scale denoise_action from [0,1] to [1, max_denoise_steps]
            steps = int(np.clip(denoise_action * cfg["max_denoise_steps"], 1, cfg["max_denoise_steps"]))
            mem_i = self._compute_memory(user)
            lat_i, flops_i = self._compute_latency(user, steps)
            qos_i = self._compute_qos(steps)
            comm_i = user.image_size + user.prompt_size
            price_i = self._estimate_price(mem_i, flops_i, comm_i)

            # QoS penalty chỉ phạt nếu bạn thực sự muốn (hiện tại code gốc phạt khi qos > yêu cầu)
            qos_penalty = 0.0
            if qos_i > user.qos_required:
                qos_penalty = cfg["lambda_qos"] * cfg["penalty_reduction_factor"] * (qos_i - user.qos_required)

            # "giá trị/giây" để ưu tiên lập lịch
            value_density = price_i / max(lat_i, 1e-9)
            candidates.append({
                "idx": i, "mem": mem_i, "lat": lat_i, "flops": flops_i,
                "price": price_i, "qos_pen": qos_penalty, "steps": steps,
                "qos_achieved": qos_i, "qos_required": user.qos_required,
                "value_density": value_density
            })

        # 2) Sắp xếp theo "giá trị/giây" giảm dần (cũng có thể sort theo price, hoặc mem nhỏ,...)
        candidates.sort(key=lambda x: x["value_density"], reverse=True)

        # 3) Gom batch greedy theo Mmax, chạy batch nối tiếp cho tới khi hết thời gian
        batches = []
        batch = []
        batch_mem = 0.0
        for c in candidates:
            if batch_mem + c["mem"] <= Mmax:
                batch.append(c)
                batch_mem += c["mem"]
            else:
                # chốt batch hiện tại (nếu có), mở batch mới
                if batch:
                    batches.append(batch)
                batch = [c]
                batch_mem = c["mem"]
        if batch:
            batches.append(batch)

        # 4) Thực thi các batch trong time_budget
        time_spent = 0.0
        served_users, rejected_users = [], []
        total_price = 0.0
        total_qos_pen = 0.0
        total_flops = 0.0
        peak_mem = 0.0
        peak_users = 0  # Thêm biến peak_users
        total_service_bonus = 0.0
        
        # QoS và diffusion steps tracking
        served_qos_achieved = []
        served_qos_required = []
        served_diffusion_steps = []
        qos_violations_count = 0

        for b in batches:
            b_mem = sum(x["mem"] for x in b)
            b_lat = max(x["lat"] for x in b)  # Thời gian batch = max latency trong batch
            b_users = len(b)  # Số user trong batch này
            peak_mem = max(peak_mem, b_mem)   # Ghi lại peak memory
            peak_users = max(peak_users, b_users)  # Ghi lại peak users per batch

            # nếu chạy batch này vẫn trong ngân sách thời gian, ta phục vụ batch
            if time_spent + b_lat <= time_budget and b_mem <= Mmax:
                time_spent += b_lat
                for x in b:
                    total_price += x["price"]
                    total_qos_pen += x["qos_pen"]
                    total_flops += x["flops"]
                    total_service_bonus += cfg["user_service_bonus"]
                    served_users.append(x["idx"])
                    
                    # Collect QoS và diffusion steps data
                    user = self.users[x["idx"]]
                    served_qos_achieved.append(x["qos_achieved"])
                    served_qos_required.append(x["qos_required"])
                    served_diffusion_steps.append(x["steps"])
                    
                    # Đếm QoS violations (QoS achieved > required)
                    if x["qos_achieved"] > x["qos_required"]:
                        qos_violations_count += 1
            else:
                # không đủ thời gian hoặc vượt Mmax, bỏ các user còn lại
                for x in b:
                    rejected_users.append(x["idx"])

        # 5) Penalties & optional system bonus
        total_penalty = 0.0

        # Latency/system time penalty (nếu muốn răng hơn, tăng lambda_latency)
        if time_spent > time_budget:
            total_penalty += cfg["lambda_latency"] * cfg["penalty_reduction_factor"] * (time_spent - time_budget)

        # FLOPs cap penalty (giữ nguyên)
        if total_flops > cfg["Gmax"]:
            total_penalty += cfg["lambda_flops"] * cfg["penalty_reduction_factor"] * (total_flops - cfg["Gmax"])

        # Peak memory penalty (theo batch)
        if peak_mem > Mmax:
            total_penalty += cfg["lambda_mem"] * cfg["penalty_reduction_factor"] * (peak_mem - Mmax)
        

        # Cộng QoS penalty (nếu có)
        total_penalty += total_qos_pen

        # ψ bonus: nếu muốn thưởng khi sử dụng thời gian hiệu quả & phục vụ đủ tỉ lệ
        service_ratio = len(served_users) / cfg["num_users"] if cfg["num_users"] > 0 else 0.0
        bonus = 0.0
        if (time_spent <= time_budget and peak_mem <= Mmax and total_flops <= cfg["Gmax"]
            and service_ratio >= cfg["lambda_serve_complete"]):
            bonus = cfg["psi"]

        final_reward = (total_price + total_service_bonus) - total_penalty + bonus

        # Tính toán QoS và diffusion steps metrics
        avg_qos_achieved = np.mean(served_qos_achieved) if served_qos_achieved else 0
        avg_qos_required = np.mean(served_qos_required) if served_qos_required else 0
        qos_satisfaction_rate = (len(served_qos_achieved) - qos_violations_count) / max(len(served_qos_achieved), 1)
        
        avg_diffusion_steps = np.mean(served_diffusion_steps) if served_diffusion_steps else 0
        min_diffusion_steps = np.min(served_diffusion_steps) if served_diffusion_steps else 0
        max_diffusion_steps = np.max(served_diffusion_steps) if served_diffusion_steps else 0

        info = {
            "served": len(served_users),
            "served_users": served_users,
            "rejected_users": rejected_users,
            "time_spent": time_spent,
            "time_budget": time_budget,
            "num_batches": len(batches),
            "peak_mem": peak_mem,
            "peak_users": peak_users,  # Thêm vào info
            "total_flops": total_flops,
            "price": total_price,
            "service_bonus": total_service_bonus,
            "penalty": total_penalty,
            "bonus": bonus,
            "service_ratio": service_ratio,
            "memory_utilization": peak_mem / Mmax if Mmax > 0 else 0,
            # QoS metrics
            "avg_qos_achieved": avg_qos_achieved,
            "avg_qos_required": avg_qos_required,
            "qos_satisfaction_rate": qos_satisfaction_rate,
            "qos_violations": qos_violations_count,
            "served_qos_achieved": served_qos_achieved,
            "served_qos_required": served_qos_required,
            # Diffusion steps metrics
            "avg_diffusion_steps": avg_diffusion_steps,
            "min_diffusion_steps": min_diffusion_steps,
            "max_diffusion_steps": max_diffusion_steps,
            "served_diffusion_steps": served_diffusion_steps,
        }
        return final_reward, info

    def _compute_reward_baseline(self, action):
        """
        BASELINE REWARD: Đơn giản, ít bonus/penalty chồng chéo
        - Chủ yếu dựa vào price reward (main signal)
        - Penalty nhẹ cho vi phạm constraints
        - Ít bonus phức tạp
        """
        total_reward = 0
        total_penalty = 0
        total_latency = 0
        total_flops = 0
        total_mem = 0
        served = 0
        info = {}

        current_memory = 0
        served_users = []
        rejected_users = []
        
        # Process all users
        for i, user in enumerate(self.users):
            serve_decision = action[2 * i]
            denoise_action = action[2 * i + 1]
            
            serve = 1 if serve_decision >= 0.5 else 0
            
            if serve:
                # Scale denoise_action from [0,1] to [1, max_denoise_steps]
                denoise_steps = int(np.clip(denoise_action * self.config["max_denoise_steps"], 1, self.config["max_denoise_steps"]))
                
                mem = self._compute_memory(user)
                latency, flops = self._compute_latency(user, denoise_steps)
                
                # Check memory constraint
                if current_memory + mem <= self.config["Mmax"]:
                    qos = self._compute_qos(denoise_steps)
                    price = self._compute_price(mem, flops, user.image_size + user.prompt_size)
                    
                    # MAIN REWARD: Price (dominant signal)
                    total_reward += price
                    
                    # SIMPLE BONUS: Small service bonus
                    total_reward += self.config["user_service_bonus"]
                    
                    # LIGHT PENALTIES: Individual constraint violations
                    if qos > user.qos_required:
                        total_penalty += self.config["lambda_qos"] * self.config["penalty_reduction_factor"] * (qos - user.qos_required)
                    
                    if latency > self.config["sys_tau"]:
                        total_penalty += self.config["lambda_latency"] * self.config["penalty_reduction_factor"] * (latency - self.config["sys_tau"])

                    total_latency += latency
                    total_flops += flops
                    current_memory += mem
                    served += 1
                    served_users.append(i)
                else:
                    rejected_users.append(i)

        total_mem = current_memory

        # LIGHT SYSTEM PENALTIES: Chỉ phạt khi vi phạm nghiêm trọng
        if total_latency > self.config["sys_tau"] * self.config["num_users"]:
            total_penalty += self.config["lambda_latency"] * self.config["penalty_reduction_factor"] * (total_latency - self.config["sys_tau"] * self.config["num_users"])
        
        if total_flops > self.config["Gmax"]:
            total_penalty += self.config["lambda_flops"] * self.config["penalty_reduction_factor"] * (total_flops - self.config["Gmax"])
        
        if total_mem > self.config["Mmax"]:
            total_penalty += self.config["lambda_mem"] * self.config["penalty_reduction_factor"] * (total_mem - self.config["Mmax"])

        # SIMPLE SYSTEM BONUS: Chỉ khi đạt tất cả constraints + phục vụ đủ user
        service_ratio = served / self.config["num_users"] if self.config["num_users"] > 0 else 0
        bonus = 0
        
        if (total_latency <= self.config["sys_tau"] * self.config["num_users"] and 
            total_flops <= self.config["Gmax"] and 
            total_mem <= self.config["Mmax"] and
            service_ratio >= self.config["lambda_serve_complete"]):
            bonus = self.config["psi"]

        # Store metrics
        info['served'] = served
        info['total_latency'] = total_latency
        info['total_flops'] = total_flops
        info['total_mem'] = total_mem
        info['bonus'] = bonus
        info['penalty'] = total_penalty
        info['memory_utilization'] = total_mem / self.config["Mmax"] if self.config["Mmax"] > 0 else 0
        info['service_ratio'] = service_ratio
        info['served_users'] = served_users
        info['rejected_users'] = rejected_users

        final_reward = total_reward - total_penalty + bonus
        return final_reward, info

    # Metrics-related methods
    def get_episode_statistics(self):
        """Lấy thống kê episode từ metrics logger"""
        if not self.enable_metrics or not self.metrics_logger:
            return {}
        return self.metrics_logger.get_episode_statistics()
    
    def get_step_statistics(self):
        """Lấy thống kê step-level từ metrics logger"""
        if not self.enable_metrics or not self.metrics_logger:
            return {}
        return self.metrics_logger.get_step_statistics()
    
    def get_metrics_summary(self, detailed=False):
        """Lấy tóm tắt metrics hiện tại"""
        if not self.enable_metrics or not self.metrics_logger:
            print("Metrics logging is disabled")
            return {}
        
        self.metrics_logger.print_summary(detailed=detailed)
        return self.metrics_logger.get_episode_statistics()
    
    def get_performance_trends(self):
        """Lấy xu hướng performance"""
        if not self.enable_metrics or not self.metrics_logger:
            return {}
        return self.metrics_logger.get_performance_trends()
    
    def get_violation_stats(self):
        """Lấy thống kê vi phạm constraints"""
        if not self.enable_metrics or not self.metrics_logger:
            return {}
        return self.metrics_logger.get_violation_statistics()
    
    def export_metrics(self, filename):
        """Xuất metrics ra file"""
        if not self.enable_metrics or not self.metrics_logger:
            print("Metrics logging is disabled")
            return
        self.metrics_logger.export_detailed_logs(filename)
    
    def reset_metrics(self):
        """Reset metrics"""
        if self.enable_metrics and self.metrics_logger:
            self.metrics_logger.reset_metrics()


if __name__ == "__main__":
    config = EnvConfig_v1_baseline("GAIServiceEnv_baseline")
    
    # Test với metrics logging enabled
    env = GAIServiceEnv_v1_baseline(config, enable_metrics=True, log_file="test_metrics.log")
    
    print("=== Testing Environment with MetricsLogger ===")
    
    # Run multiple episodes to test metrics
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        state = env.reset()
        episode_reward = 0
        
        for step in range(config["T"]):
            action = np.random.uniform(0, 1, size=2 * config["num_users"])
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if step == 0:  # Print details for first step
                print(f"Step {step}: Reward={reward:.4f}, Served={info['served']}")
                print(f"  Time: {info['time_spent']:.3f}s/{info['time_budget']}s")
                print(f"  Memory: {info['peak_mem']:.2f}/{config['Mmax']} GB")
                print(f"  Batches: {info['num_batches']}, Peak users/batch: {info['peak_users']}")
                print(f"  QoS: Achieved={info['avg_qos_achieved']:.2f}, Required={info['avg_qos_required']:.2f}, Satisfaction={info['qos_satisfaction_rate']:.2%}")
                print(f"  Diffusion Steps: Avg={info['avg_diffusion_steps']:.1f}, Range=[{info['min_diffusion_steps']:.0f}-{info['max_diffusion_steps']:.0f}]")
            
            if done:
                break
        
        print(f"Episode {episode + 1} completed: Total reward = {episode_reward:.4f}")
    
    # Test metrics summary
    print("\n=== Metrics Summary ===")
    env.get_metrics_summary(detailed=True)
    
    # Test performance trends
    trends = env.get_performance_trends()
    if trends:
        print(f"\nReward trend (last episodes): {[f'{r:.3f}' for r in trends['reward_trend']]}")
    
    # Test violation statistics
    violations = env.get_violation_stats()
    if violations:
        print(f"\nViolation rates: Memory={violations['memory_violation_rate']:.2%}, "
              f"Time={violations['time_violation_rate']:.2%}")
    
    # Test export functionality
    print("\nExporting metrics to 'test_detailed_metrics.json'...")
    env.export_metrics("test_detailed_metrics.json")
    
    print("\n=== Testing Single Step (Original Test) ===")
    state = env.reset()
    print("Initial State shape:", state.shape)
    action = np.random.uniform(0, 1, size=2 * config["num_users"])
    next_state, reward, done, info = env.step(action)
    print("Next State shape:", next_state.shape, "Reward:", reward, "Done:", done)
    
    # Test parallel batching details
    print("\n=== Parallel Batch Processing Details ===")
    print(f"Time budget: {info['time_budget']}s")
    print(f"Time spent: {info['time_spent']:.3f}s")
    print(f"Number of batches: {info['num_batches']}")
    print(f"Peak memory: {info['peak_mem']:.2f} GB (limit: {config['Mmax']} GB)")
    print(f"Peak users per batch: {info['peak_users']} users")
    print(f"Served users: {info['served']}/{config['num_users']}")
    print(f"Service ratio: {info['service_ratio']:.2f}")
    print(f"Total price: {info['price']:.6f}")
    print(f"QoS Performance: Achieved={info['avg_qos_achieved']:.2f}, Required={info['avg_qos_required']:.2f}")
    print(f"QoS Satisfaction Rate: {info['qos_satisfaction_rate']:.2%} ({info['qos_violations']} violations)")
    print(f"Diffusion Steps: Avg={info['avg_diffusion_steps']:.1f}, Range=[{info['min_diffusion_steps']:.0f}-{info['max_diffusion_steps']:.0f}]")
    if info['served_diffusion_steps']:
        print(f"Individual Diffusion Steps: {info['served_diffusion_steps']}")
        print(f"Individual QoS Achieved: {[f'{q:.1f}' for q in info['served_qos_achieved']]}")
        print(f"Individual QoS Required: {[f'{q:.1f}' for q in info['served_qos_required']]}")
    
    print("\n=== Testing Environment without Metrics ===")
    env_no_metrics = GAIServiceEnv_v1_baseline(config, enable_metrics=False)
    state = env_no_metrics.reset()
    action = np.random.uniform(0, 1, size=2 * config["num_users"])
    next_state, reward, done, info = env_no_metrics.step(action)
    print("Environment without metrics works correctly")
    print(f"Reward: {reward:.4f}, Served: {info['served']}")
    print(f"QoS: Achieved={info['avg_qos_achieved']:.2f}, Satisfaction={info['qos_satisfaction_rate']:.2%}")
    print(f"Diffusion Steps: Avg={info['avg_diffusion_steps']:.1f}")
    
    print("\n=== MetricsLogger Test Completed ===")
    print("Check 'test_metrics.log' and 'test_detailed_metrics.json' for exported data")