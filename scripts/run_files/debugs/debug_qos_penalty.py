import numpy as np
from env.env import GAIServiceEnv_v1, EnvConfig_v1

def analyze_qos_penalty_issue():
    """Analyze QoS penalty calculation in detail"""
    config = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(config)
    env.reset()
    
    print("=== QoS PENALTY ANALYSIS ===")
    print(f"Lambda QoS: {config['lambda_qos']}")
    print(f"Number of users: {config['num_users']}")
    
    # Check user QoS requirements
    print("\n--- User QoS Requirements ---")
    for i, user in enumerate(env.users):
        print(f"User {i}: QoS requirement = {user.qos_required:.2f}")
    
    # Test QoS computation with fixed denoise steps
    print(f"\n--- QoS Computation (multiple runs for same denoise steps) ---")
    for denoise_steps in [1, 5, 10]:
        print(f"\nDenoise steps: {denoise_steps}")
        qos_values = []
        for run in range(5):
            qos = env._compute_qos(denoise_steps)
            qos_values.append(qos)
            print(f"  Run {run+1}: QoS = {qos:.2f}")
        print(f"  Mean: {np.mean(qos_values):.2f}, Std: {np.std(qos_values):.2f}")
    
    # Calculate penalty for different scenarios
    print(f"\n--- Penalty Calculation Analysis ---")
    
    # Test với denoise_steps = 1 (worst case from debug)
    denoise_steps = 1
    total_qos_penalty = 0
    
    print(f"\nTesting denoise_steps = {denoise_steps}:")
    for i, user in enumerate(env.users):
        qos = env._compute_qos(denoise_steps)
        penalty = config["lambda_qos"] * max(0, qos - user.qos_required)
        total_qos_penalty += penalty
        
        print(f"User {i}: QoS={qos:.2f}, Req={user.qos_required:.2f}, Penalty={penalty:.2f}")
    
    print(f"Total QoS penalty: {total_qos_penalty:.2f}")
    
    # Compare with actual environment calculation
    print(f"\n--- Environment vs Manual Calculation ---")
    action = np.array([1, 1] * config["num_users"])  # Serve all with 1 denoise step
    
    env.reset()
    obs, env_reward, done, info = env.step(action)
    
    print(f"Environment reward: {env_reward:.2f}")
    
    # Manual calculation
    manual_components = calculate_reward_components_manual(env, action)
    manual_reward = manual_components['total_reward'] - manual_components['total_penalty'] + manual_components['bonus']
    
    print(f"Manual reward: {manual_reward:.2f}")
    print(f"Manual QoS penalty: {manual_components['qos_penalty']:.2f}")
    print(f"Difference: {abs(env_reward - manual_reward):.2f}")

def calculate_reward_components_manual(env, action):
    """Calculate reward components manually for verification"""
    total_reward = 0
    qos_penalty = 0
    latency_penalty = 0
    resource_penalty = 0
    total_latency = 0
    total_flops = 0
    total_mem = 0
    
    for i, user in enumerate(env.users):
        serve = int(round(action[2 * i]))
        denoise_steps = int(round(action[2 * i + 1]))
        denoise_steps = np.clip(denoise_steps, 1, env.config["max_denoise_steps"])

        if serve:
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
        'bonus': bonus
    }

if __name__ == "__main__":
    analyze_qos_penalty_issue()
