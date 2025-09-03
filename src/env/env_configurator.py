"""
Configuration manager cho các phiên bản environment khác nhau
"""

import numpy as np


class EnvironmentConfigurator:
    """
    Quản lý cấu hình cho các phiên bản environment khác nhau
    """
    
    @staticmethod
    def get_config(config_type="original", env_name="GAIServiceEnv"):
        """
        Lấy cấu hình theo loại
        
        Args:
            config_type: "original", "improved", "baseline"
            env_name: tên environment
        """
        print(f"Using {config_type} configuration for: {env_name}")
        
        if config_type == "baseline":
            return EnvironmentConfigurator._get_baseline_config()
        elif config_type == "improved":
            return EnvironmentConfigurator._get_improved_config()
        else:
            return EnvironmentConfigurator._get_original_config()
    
    @staticmethod
    def _get_original_config():
        """Cấu hình gốc với các vấn đề đã được chỉ ra"""
        return {
            "num_users": 10,
            "T": 10,
            "sys_tau": 0.5,
            "Gmax": 1e10,
            "Mmax": 48,
            "lambda_qos": 0.75,
            "lambda_latency": 0.5,
            "lambda_mem": 1.0,
            "lambda_flops": 1.0,
            "lambda_serve_complete": 1.0,  # Vấn đề: comment nói 80% nhưng set 1.0
            "PVM": 1e12,
            "Rmem": 2.304e12,
            "max_denoise_steps": 35,
            "c1": 3.81e-6,
            "c2": 4.86,  # Vấn đề: c2 quá lớn
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
            "user_service_bonus": 3,
            "memory_utilization_bonus": 5,
            "fairness_bonus": 3,
            "min_utilization_threshold": 0.55,
            "penalty_reduction_factor": 0.75,  # Vấn đề: khá cứng
            "efficiency_bonus_scale": 0.1,
            "service_quality_weight": 0.05,
        }
    
    @staticmethod
    def _get_improved_config():
        """Cấu hình đã cải thiện các vấn đề được chỉ ra"""
        return {
            "num_users": 10,
            "T": 10,
            "sys_tau": 0.5,
            "Gmax": 1e10,
            "Mmax": 96,  # Tăng từ 48
            "lambda_qos": 0.75,
            "lambda_latency": 0.5,
            "lambda_mem": 1.0,
            "lambda_flops": 1.0,
            "lambda_serve_complete": 0.8,  # Sửa từ 1.0 thành 0.8
            "PVM": 1e12,
            "Rmem": 2.304e12,
            "max_denoise_steps": 35,
            "c1": 3.81e-6,
            "c2": 2.0,  # Giảm từ 4.86
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
            "user_service_bonus": 3,
            "memory_utilization_bonus": 5,
            "fairness_bonus": 3,
            "min_utilization_threshold": 0.55,
            "penalty_reduction_factor": 0.5,  # Giảm từ 0.75
            "efficiency_bonus_scale": 0.1,
            "service_quality_weight": 0.05,
        }
    
    @staticmethod
    def _get_baseline_config():
        """Cấu hình baseline đơn giản để agent học ổn định"""
        return {
            "num_users": 10,
            "T": 10,
            "sys_tau": 0.5,
            "Gmax": 1e10,
            "Mmax": 96,  # Tăng để tránh memory bottleneck
            "lambda_qos": 0.3,  # Giảm penalty
            "lambda_latency": 0.3,  # Giảm penalty
            "lambda_mem": 0.5,  # Giảm penalty
            "lambda_flops": 0.5,  # Giảm penalty
            "lambda_serve_complete": 0.8,
            "PVM": 1e12,
            "Rmem": 2.304e12,
            "max_denoise_steps": 35,
            "c1": 3.81e-6,
            "c2": 2.0,  # Giảm c2
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
            "psi": 50,  # Giảm system bonus
            "user_service_bonus": 1,  # Đơn giản hóa
            "penalty_reduction_factor": 0.3,  # Penalty rất nhẹ
        }
    
    @staticmethod
    def print_config_comparison():
        """In so sánh các cấu hình"""
        configs = {
            "Original": EnvironmentConfigurator._get_original_config(),
            "Improved": EnvironmentConfigurator._get_improved_config(),
            "Baseline": EnvironmentConfigurator._get_baseline_config()
        }
        
        print("=" * 80)
        print("ENVIRONMENT CONFIGURATION COMPARISON")
        print("=" * 80)
        
        # Các thông số quan trọng để so sánh
        key_params = [
            "lambda_serve_complete", "Mmax", "c2", "penalty_reduction_factor",
            "lambda_qos", "lambda_latency", "psi", "user_service_bonus"
        ]
        
        for param in key_params:
            print(f"\n{param}:")
            for config_name, config in configs.items():
                value = config.get(param, "N/A")
                print(f"  {config_name:>12}: {value}")
        
        print("\n" + "=" * 80)
        print("KEY IMPROVEMENTS:")
        print("- lambda_serve_complete: 1.0 → 0.8 (khả thi hơn)")
        print("- Mmax: 48 → 96 (tránh memory bottleneck)")  
        print("- c2: 4.86 → 2.0 (cân bằng compute vs memory)")
        print("- penalty_reduction_factor: 0.75 → 0.5/0.3 (nhẹ tay hơn)")
        print("- Baseline có penalty nhẹ nhất để agent học ổn định")
        print("=" * 80)


if __name__ == "__main__":
    # Demo so sánh cấu hình
    EnvironmentConfigurator.print_config_comparison()
    
    # Test lấy cấu hình
    baseline_config = EnvironmentConfigurator.get_config("baseline")
    print(f"\nBaseline config lambda_serve_complete: {baseline_config['lambda_serve_complete']}")
