import numpy as np
from env.env import GAIServiceEnv_v1, EnvConfig_v1

def calculate_manual_reward_components(env, action):
    """Manually calculate reward components to debug negative rewards"""
    total_reward = 0
    total_penalty = 0
    qos_penalty = 0 
    latency_penalty = 0
    resource_penalty = 0
    total_latency = 0
    total_flops = 0
    total_mem = 0
    total_served = 0
    
    for i, user in enumerate(env.users):
        serve = int(round(action[2 * i]))
        denoise_steps = int(round(action[2 * i + 1]))
        denoise_steps = np.clip(denoise_steps, 1, env.config["max_denoise_steps"])

        if serve:
            total_served += 1
            latency, flops = env._compute_latency(user, denoise_steps)
            mem = env._compute_memory(user)
            qos = env._compute_qos(denoise_steps)
            price = env._compute_price(mem, flops, user.image_size + user.prompt_size)
            penalty_q = env.config["lambda_qos"] * max(0, qos - user.qos_required)
            penalty_l = env.config["lambda_latency"] * max(0, latency - env.config["sys_tau"])

            total_reward += price
            qos_penalty += penalty_q
            latency_penalty += penalty_l
            total_latency += latency
            total_flops += flops
            total_mem += mem

    # System-level penalties
    if total_latency > env.config["sys_tau"] * env.config["num_users"]:
        sys_latency_penalty = env.config["lambda_latency"] * (total_latency - env.config["sys_tau"] * env.config["num_users"])
        latency_penalty += sys_latency_penalty
        
    if total_flops > env.config["Gmax"]:
        flops_penalty = env.config["lambda_flops"] * (total_flops - env.config["Gmax"])
        resource_penalty += flops_penalty
        
    if total_mem > env.config["Mmax"]:
        mem_penalty = env.config["lambda_mem"] * (total_mem - env.config["Mmax"])
        resource_penalty += mem_penalty

    total_penalty = qos_penalty + latency_penalty + resource_penalty

    # Bonus
    bonus = 0
    constraints_met = (
        total_latency <= env.config["sys_tau"] * env.config["num_users"] and 
        total_flops <= env.config["Gmax"] and 
        total_mem <= env.config["Mmax"]
    )
    if constraints_met:
        bonus = env.config["psi"]

    return {
        'total_reward': total_reward,
        'total_penalty': total_penalty,
        'qos_penalty': qos_penalty,
        'latency_penalty': latency_penalty,
        'resource_penalty': resource_penalty,
        'bonus': bonus,
        'total_served': total_served,
        'total_latency': total_latency,
        'total_flops': total_flops,
        'total_mem': total_mem,
        'constraints_met': constraints_met
    }

def debug_reward_components():
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    
    print("=== DEBUGGING REWARD COMPONENTS ===")
    print("Config values:")
    print(f"  lambda_qos: {config['lambda_qos']}")
    print(f"  lambda_latency: {config['lambda_latency']}")  
    print(f"  lambda_mem: {config['lambda_mem']}")
    print(f"  lambda_flops: {config['lambda_flops']}")
    print(f"  psi (bonus): {config['psi']}")
    print(f"  sys_tau: {config['sys_tau']}")
    print(f"  Gmax: {config['Gmax']}")
    print(f"  Mmax: {config['Mmax']}")
    print(f"  num_users: {config['num_users']}")
    
    # Test với một số actions khác nhau (20 elements for 10 users)
    test_actions = [
        np.array([1, 10] * config["num_users"]),  # Serve all với denoise 10
        np.array([0, 1] * config["num_users"]),   # No serving
        np.array([1, 1] * config["num_users"]),   # Min denoise for all
        np.array([1, 5] * config["num_users"])    # Medium denoise for all
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n--- Test Action {i+1} ---")
        print(f"Action pattern: {action[:4]}... (first 2 users)")
        
        env.reset()
        obs, reward, done, info = env.step(action)
        
        print(f"Final Reward: {reward:.2f}")
        
        # Calculate manual components
        components = calculate_manual_reward_components(env, action)
        
        print("Components:")
        print(f"  Raw reward: {components['total_reward']:.6f}")
        print(f"  Total penalty: {components['total_penalty']:.6f}")
        print(f"  Bonus: {components['bonus']:.2f}")
        print(f"  QoS penalty: {components['qos_penalty']:.6f}")
        print(f"  Latency penalty: {components['latency_penalty']:.6f}")
        print(f"  Resource penalty: {components['resource_penalty']:.6f}")
        print(f"  Total served: {components['total_served']}")
        print(f"  Total latency: {components['total_latency']:.2f}")
        print(f"  Total FLOPS: {components['total_flops']:.2e}")
        print(f"  Total memory: {components['total_mem']:.2e}")
        
        # Check constraints
        tau_limit = config["sys_tau"] * config["num_users"]
        print(f"  Latency constraint: {components['total_latency']:.2f} <= {tau_limit} ? {components['total_latency'] <= tau_limit}")
        print(f"  FLOPS constraint: {components['total_flops']:.2e} <= {config['Gmax']:.2e} ? {components['total_flops'] <= config['Gmax']}")
        print(f"  Memory constraint: {components['total_mem']:.2e} <= {config['Mmax']:.2e} ? {components['total_mem'] <= config['Mmax']}")
        
        # Analyze why negative
        if reward < 0:
            print(f"  ❌ NEGATIVE REWARD!")
            print(f"    Raw reward: {components['total_reward']:.6f}")
            print(f"    Total penalty: {components['total_penalty']:.6f}")
            print(f"    Difference: {components['total_reward'] - components['total_penalty']:.6f}")
            if components['total_reward'] < 1e-5:
                print(f"    ⚠️  Raw reward is very small - check pricing coefficients!")
        else:
            print(f"  ✅ Positive reward")
            
        # Verify manual calculation matches environment
        manual_final = components['total_reward'] - components['total_penalty'] + components['bonus']
        print(f"Manual calculation: {manual_final:.6f}")
        print(f"Environment reward: {reward:.6f}")
        print(f"Difference: {abs(manual_final - reward):.8f}")

    # Analyze typical prices and penalties
    print(f"\n=== PRICE vs PENALTY ANALYSIS ===")
    env.reset()
    sample_user = env.users[0] 
    
    print(f"Sample user QoS requirement: {sample_user.qos_required:.2f}")
    print(f"Sample user image size: {sample_user.image_size}")
    print(f"Sample user prompt size: {sample_user.prompt_size}")
    
    for denoise_steps in [1, 5, 10, 15]:
        latency, flops = env._compute_latency(sample_user, denoise_steps)
        mem = env._compute_memory(sample_user)
        qos = env._compute_qos(denoise_steps)
        price = env._compute_price(mem, flops, sample_user.image_size + sample_user.prompt_size)
        
        penalty_q = config["lambda_qos"] * max(0, qos - sample_user.qos_required)
        penalty_l = config["lambda_latency"] * max(0, latency - config["sys_tau"])
        
        print(f"Denoise steps {denoise_steps:2d}: price={price:.6f}, qos_penalty={penalty_q:.6f}, latency_penalty={penalty_l:.6f}, qos={qos:.2f}, latency={latency:.2f}")

if __name__ == "__main__":
    debug_reward_components()
