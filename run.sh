# python main.py --env_name gail-service-env --algo ql  
# python main.py --env_name gail-service-env --algo gdql 
# python main.py --env_name gail-service-env --algo ppo 
# python main.py --env_name gail-service-env --algo gppo 
# python main.py --env_name gail-service-env --algo bc 

# tmux new-session -s "rl" -n "ql" "python main.py --env_name gail-service-env --algo ql"
# tmux new-window -t "rl" -n "gdql" "python main.py --env_name gail-service-env --algo gdql"
# tmux new-window -t "rl" -n "ppo" "python main.py --env_name gail-service-env --algo ppo"
# tmux new-window -t "rl" -n "gppo" "python main.py --env_name gail-service-env --algo gppo"
# tmux new-window -t "rl" -n "bc" "python main.py --env_name gail-service-env --algo bc"
tmux kill-session -t "mrl"

# tmux new-session -s "mrl" 
# tmux new-window -t "mrl" -n "ql" "python test.py --algo gdql"
# tmux new-window -t "mrl" -n "gdql" "python test.py --algo dql"
# tmux new-window -t "mrl" -n "ppo" "python test.py --algo ppo"
# tmux new-window -t "mrl" -n "gppo" "python test.py --algo gppo"
# tmux new-window -t "mrl" -n "a2c" "python test.py --algo a2c"
# tmux new-window -t "mrl" -n "da2c" "python test.py --algo da2c"
# tmux new-window -t "mrl" -n "bc" "python test.py --algo bc"

# quit tmux s
#!/bin/bash

N=10000
ALGO="ppo"
NUM_EPISODES=2000

echo "Running $ALGO sequentially with seeds 0 to $N"

# Create results directory
mkdir -p results

for seed in $(seq 1853 $N); do
    echo "🏃 Running seed $seed/$N..."
    
    # Run experiment
    python test.py --algo $ALGO --seed $seed --num_episodes $NUM_EPISODES
    
    # Move results to seed-specific directory
    # seed_dir="results/seed_${seed}"
    # mkdir -p "$seed_dir"
    
    if [ -f "test/episode_rewards_${ALGO}.npy" ]; then
        mv "test/episode_rewards_${ALGO}.npy" "$seed_dir/"
        echo "✓ Saved rewards for seed $seed"
    fi
    
    if [ -f "test/loss_log_${ALGO}.json" ]; then
        mv "test/loss_log_${ALGO}.json" "$seed_dir/"
        echo "✓ Saved losses for seed $seed"
    fi
    
    echo "✅ Completed seed $seed"
    echo ""
done

echo "🎉 All seeds completed! Results in results/ directory"

# Generate summary
python3 << EOF
import numpy as np
import os

seeds = list(range(0, $N + 1))
final_rewards = []

print("📊 Summary:")
print("Seed | Final Reward (last 100 episodes)")
print("-" * 40)

for seed in seeds:
    try:
        rewards = np.load(f"results/seed_{seed}/episode_rewards_${ALGO}.npy")
        final_reward = rewards[-100:].mean()
        final_rewards.append(final_reward)
        print(f"{seed:4d} | {final_reward:8.2f}")
    except:
        print(f"{seed:4d} | Failed")

if final_rewards:
    print("-" * 40)
    print(f"Mean | {np.mean(final_rewards):8.2f}")
    print(f"Std  | {np.std(final_rewards):8.2f}")
    print(f"Best | {np.max(final_rewards):8.2f}")
    print(f"Worst| {np.min(final_rewards):8.2f}")
    print("📈 Summary statistics calculated.")
else:
    print("No valid rewards found to summarize.")
EOF