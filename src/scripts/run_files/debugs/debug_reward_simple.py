#!/usr/bin/env python3

import numpy as np
from env.env import EnvConfig_v1, GAIServiceEnv_v1

def debug_reward_components():
    """Debug reward components để tìm nguyên nhân reward âm"""
    
    # Tạo environment
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    
    print("=== DEBUGGING REWARD COMPONENTS ===")
    print(f"Config values:")
    print(f"  lambda_qos: {config['lambda_qos']}")
    print(f"  lambda_latency: {config['lambda_latency']}")
    print(f"  lambda_mem: {config['lambda_mem']}")
    print(f"  lambda_flops: {config['lambda_flops']}")
    print(f"  psi (bonus): {config['psi']}")
    print(f"  sys_tau: {config['sys_tau']}")
    print(f"  Gmax: {config['Gmax']}")
    print(f"  Mmax: {config['Mmax']}")
    print()
    
    # Reset environment
    state = env.reset()
    
    # Test với different actions
    test_actions = [
        # [serve_user_0, denoise_steps_0, serve_user_1, denoise_steps_1, ...]
        np.array([1, 10] * config["num_users"]),  # Serve all users with 10 steps
        np.array([1, 5] * config["num_users"]),   # Serve all users with 5 steps
        np.array([0, 1] * config["num_users"]),   # Serve no users
        np.array([1, 1] * config["num_users"]),   # Serve all users with min steps
        np.array([1, 50] * config["num_users"]),  # Serve all users with max steps
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Test Action {i+1} ---")
        print(f"Action pattern: {action[:4]}... (first 2 users)")
        
        reward, info = env._compute_reward(action)
        
        print(f"Final Reward: {reward:.2f}")
        print(f"Components:")
        print(f"  Raw reward: {info['total_reward_raw']:.2f}")
        print(f"  Total penalty: {info['total_penalty']:.2f}")
        print(f"  Bonus: {info['bonus']:.2f}")
        print(f"  Served users: {info['served_users']}")
        print(f"  Constraints met: {info['constraints_met']}")
        print(f"  Penalty breakdown:")
        print(f"    QoS penalty: {info['penalty_breakdown']['qos_penalty']:.2f}")
        print(f"    Latency penalty: {info['penalty_breakdown']['latency_penalty']:.2f}")
        print(f"    Resource penalty: {info['penalty_breakdown']['resource_penalty']:.2f}")
        print(f"  Resource usage:")
        print(f"    Total latency: {info['total_latency']:.2f}")
        print(f"    Total FLOPs: {info['total_flops']:.2e}")
        print(f"    Total memory: {info['total_mem']:.2e}")
        
        # Analyze WHY reward is negative
        print(f"  Analysis:")
        if info['total_reward_raw'] < info['total_penalty']:
            print(f"    ❌ Raw reward ({info['total_reward_raw']:.2f}) < Penalty ({info['total_penalty']:.2f})")
            if info['penalty_breakdown']['latency_penalty'] > 0:
                print(f"    ❌ High latency penalty: {info['penalty_breakdown']['latency_penalty']:.2f}")
            if info['penalty_breakdown']['qos_penalty'] > 0:
                print(f"    ❌ High QoS penalty: {info['penalty_breakdown']['qos_penalty']:.2f}")
            if info['penalty_breakdown']['resource_penalty'] > 0:
                print(f"    ❌ High resource penalty: {info['penalty_breakdown']['resource_penalty']:.2f}")
        
        if not info['constraints_met']:
            print(f"    ❌ Constraints not met - no bonus ({config['psi']})")
        
    # Test reward components in detail
    print(f"\n=== DETAILED ANALYSIS ===")
    
    # Sample user metrics
    user = env.users[0]
    denoise_steps = 10
    
    print(f"Sample user analysis:")
    print(f"  Position: {user.position}")
    print(f"  Image size: {user.image_size:.2f} bytes")
    print(f"  Prompt size: {user.prompt_size:.2f} bytes") 
    print(f"  QoS required: {user.qos_required:.2f}")
    
    # Compute individual components
    latency, flops = env._compute_latency(user, denoise_steps)
    mem = env._compute_memory(user)
    qos = env._compute_qos(denoise_steps)
    price = env._compute_price(mem, flops, user.image_size + user.prompt_size)
    
    print(f"  Computed metrics:")
    print(f"    Latency: {latency:.4f} s")
    print(f"    FLOPs: {flops:.2e}")
    print(f"    Memory: {mem:.2e}")
    print(f"    QoS score: {qos:.2f}")
    print(f"    Price: {price:.6f}")
    
    # Check if price is too small
    print(f"  Price components:")
    print(f"    Memory component: {1e-7 * mem:.6f}")
    print(f"    FLOPs component: {1e-5 * flops:.6f}")
    print(f"    Comm component: {2.5e-6 * (user.image_size + user.prompt_size):.6f}")
    
    penalty_q = config['lambda_qos'] * max(0, qos - user.qos_required)
    penalty_l = config['lambda_latency'] * max(0, latency - config['sys_tau'])
    
    print(f"  Penalties:")
    print(f"    QoS penalty: {penalty_q:.6f}")
    print(f"    Latency penalty: {penalty_l:.6f}")
    print(f"    Net per user: {price - penalty_q - penalty_l:.6f}")

if __name__ == "__main__":
    debug_reward_components()
