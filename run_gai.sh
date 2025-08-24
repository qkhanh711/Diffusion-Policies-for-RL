SEED=22
EPISODES=500
ENV_NAME="gail-service-env-v3"

tmux kill-session -t "n-gai"

tmux new-session -s "n-gai" -n "dql"  "python main.py --algo dql   --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window  -t "n-gai" -n "ppo"   "python main.py --algo dppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window  -t "n-gai" -n "gppo"  "python main.py --algo gppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window  -t "n-gai" -n "gdql"  "python main.py --algo gdql --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window  -t "n-gai" -n "a2c"   "python main.py --algo a2c  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window  -t "n-gai" -n "da2c"  "python main.py --algo da2c --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 