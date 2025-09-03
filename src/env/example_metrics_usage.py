#!/usr/bin/env python3
"""
Example script demonstrating how to use MetricsLogger with GAI Service Environment
Tích hợp MetricsLogger vào quá trình training hoặc evaluation
"""

import numpy as np
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from env_v6_baseline import GAIServiceEnv_v1_baseline, EnvConfig_v1_baseline


def run_training_example():
    """Example của việc sử dụng MetricsLogger trong training"""
    print("=== GAI Service Environment with MetricsLogger - Training Example ===\n")
    
    # 1. Setup environment with metrics
    config = EnvConfig_v1_baseline("GAIServiceEnv_Training")
    env = GAIServiceEnv_v1_baseline(
        config, 
        enable_metrics=True, 
        metrics_window_size=50,  # Track last 50 episodes
        log_file="training_metrics.log"
    )
    
    # 2. Simulate training episodes
    num_episodes = 20
    print(f"Running {num_episodes} training episodes...\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config["T"]):
            # Simple policy: random actions with some bias toward serving users
            action = np.random.uniform(0, 1, size=2 * config["num_users"])
            
            # Bias toward serving (serve_decision > 0.3)
            for i in range(config["num_users"]):
                if action[2*i] < 0.7:  # 70% chance to serve
                    action[2*i] = np.random.uniform(0.5, 1.0)
                # Reasonable denoise steps
                action[2*i + 1] = np.random.uniform(10, 25)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        # Print progress every 5 episodes
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED - METRICS SUMMARY")
    print("="*60)
    
    # 3. Get comprehensive metrics summary
    env.get_metrics_summary(detailed=True)
    
    # 4. Analyze performance trends
    print("\n🔍 PERFORMANCE ANALYSIS:")
    trends = env.get_performance_trends()
    if trends and len(trends['reward_trend']) > 1:
        recent_avg = np.mean(trends['reward_trend'][-5:])  # Last 5 episodes
        early_avg = np.mean(trends['reward_trend'][:5])    # First 5 episodes
        improvement = recent_avg - early_avg
        print(f"  Performance Improvement: {improvement:+.2f} (early: {early_avg:.2f} → recent: {recent_avg:.2f})")
        
        # Service ratio trend
        recent_service = np.mean(trends['service_ratio_trend'][-5:])
        early_service = np.mean(trends['service_ratio_trend'][:5])
        service_improvement = recent_service - early_service
        print(f"  Service Ratio Improvement: {service_improvement:+.3f} (early: {early_service:.3f} → recent: {recent_service:.3f})")
    
    # 5. Check constraint violations
    violations = env.get_violation_stats()
    print(f"\n⚠️  CONSTRAINT ADHERENCE:")
    print(f"  Memory violations: {violations['memory_violation_rate']:.1%}")
    print(f"  Time violations: {violations['time_violation_rate']:.1%}")
    print(f"  FLOPs violations: {violations['flops_violation_rate']:.1%}")
    if violations['total_violations'] == 0:
        print("  ✅ Perfect constraint adherence!")
    
    # 6. Export detailed metrics for further analysis
    env.export_metrics("training_detailed_metrics.json")
    print(f"\n📊 Detailed metrics exported to 'training_detailed_metrics.json'")
    print(f"📋 Episode logs saved to 'training_metrics.log'")
    
    return env


def run_evaluation_example():
    """Example của việc sử dụng MetricsLogger trong evaluation"""
    print("\n\n=== EVALUATION MODE EXAMPLE ===\n")
    
    config = EnvConfig_v1_baseline("GAIServiceEnv_Eval")
    env = GAIServiceEnv_v1_baseline(
        config, 
        enable_metrics=True, 
        metrics_window_size=10,  # Smaller window for evaluation
        log_file="evaluation_metrics.log"
    )
    
    num_eval_episodes = 5
    print(f"Running {num_eval_episodes} evaluation episodes with different policies...\n")
    
    policies = [
        ("Aggressive", lambda: np.random.uniform(0.8, 1.0, size=2 * config["num_users"])),
        ("Conservative", lambda: np.random.uniform(0.2, 0.6, size=2 * config["num_users"])),
        ("Balanced", lambda: np.random.uniform(0.4, 0.8, size=2 * config["num_users"])),
    ]
    
    policy_results = {}
    
    for policy_name, policy_fn in policies:
        print(f"\n--- Testing {policy_name} Policy ---")
        env.reset_metrics()  # Reset metrics for each policy
        
        for episode in range(num_eval_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(config["T"]):
                action = policy_fn()
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
        
        # Get results for this policy
        stats = env.get_episode_statistics()
        violations = env.get_violation_stats()
        
        policy_results[policy_name] = {
            'avg_reward': stats.get('avg_episode_reward', 0),
            'avg_service_ratio': stats.get('avg_service_ratio', 0),
            'violation_rate': violations.get('memory_violation_rate', 0) + violations.get('time_violation_rate', 0),
            'qos_satisfaction': env.get_step_statistics().get('qos_satisfaction_rate', {}).get('mean', 0),
            'avg_diffusion_steps': env.get_step_statistics().get('avg_diffusion_steps', {}).get('mean', 0),
        }
        
        print(f"  Average Reward: {stats.get('avg_episode_reward', 0):.2f}")
        print(f"  Service Ratio: {stats.get('avg_service_ratio', 0):.3f}")
        print(f"  Violation Rate: {policy_results[policy_name]['violation_rate']:.1%}")
        print(f"  QoS Satisfaction: {policy_results[policy_name]['qos_satisfaction']:.2%}")
        print(f"  Avg Diffusion Steps: {policy_results[policy_name]['avg_diffusion_steps']:.1f}")
    
    # Compare policies
    print(f"\n📈 POLICY COMPARISON:")
    best_policy = max(policy_results.keys(), key=lambda p: policy_results[p]['avg_reward'])
    print(f"  Best Performing Policy: {best_policy}")
    
    for policy, results in policy_results.items():
        marker = "🏆" if policy == best_policy else "  "
        print(f"  {marker} {policy:12s}: Reward={results['avg_reward']:7.2f}, "
              f"Service={results['avg_service_ratio']:.3f}, "
              f"Violations={results['violation_rate']:.1%}, "
              f"QoS={results['qos_satisfaction']:.1%}, "
              f"Steps={results['avg_diffusion_steps']:.1f}")


def run_real_time_monitoring_example():
    """Example của real-time monitoring với MetricsLogger"""
    print("\n\n=== REAL-TIME MONITORING EXAMPLE ===\n")
    
    config = EnvConfig_v1_baseline("GAIServiceEnv_Monitor")
    env = GAIServiceEnv_v1_baseline(config, enable_metrics=True, metrics_window_size=20)
    
    print("Simulating real-time environment monitoring...")
    print("Printing metrics every 3 episodes:\n")
    
    for episode in range(15):
        state = env.reset()
        
        for step in range(config["T"]):
            # Simulate a learning agent's improving policy
            base_serve_prob = 0.3 + 0.4 * (episode / 15)  # Gradually more aggressive
            action = np.random.uniform(0, 1, size=2 * config["num_users"])
            
            # Apply policy
            for i in range(config["num_users"]):
                if np.random.random() < base_serve_prob:
                    action[2*i] = np.random.uniform(0.6, 1.0)
                else:
                    action[2*i] = np.random.uniform(0, 0.4)
                action[2*i + 1] = np.random.uniform(5, 30)
            
            next_state, reward, done, info = env.step(action)
            
            if done:
                break
        
        # Print mini-summary every 3 episodes
        if (episode + 1) % 3 == 0:
            stats = env.get_episode_statistics()
            violations = env.get_violation_stats()
            print(f"Episodes {episode-1:2d}-{episode+1:2d}: "
                  f"Avg Reward = {stats.get('avg_episode_reward', 0):6.1f}, "
                  f"Service Ratio = {stats.get('avg_service_ratio', 0):.3f}, "
                  f"Violations = {violations.get('total_violations', 0):2d}")
    
    print(f"\n🔚 Monitoring completed. Final summary:")
    env.get_metrics_summary(detailed=False)


if __name__ == "__main__":
    print("🚀 MetricsLogger Usage Examples for GAI Service Environment\n")
    
    try:
        # Example 1: Training scenario
        training_env = run_training_example()
        
        # Example 2: Evaluation scenario  
        run_evaluation_example()
        
        # Example 3: Real-time monitoring
        run_real_time_monitoring_example()
        
        print(f"\n✅ All examples completed successfully!")
        print(f"📁 Check the following files for logged data:")
        print(f"   - training_metrics.log")
        print(f"   - training_detailed_metrics.json") 
        print(f"   - evaluation_metrics.log")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
