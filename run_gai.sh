SEED=24
EPISODES=500
ENV_NAME="gail-service-env"

tmux kill-session -t "gai"
tmux new-session  -s "gai" -n "watch" "watch -n 1 nvidia-smi"
tmux new-window   -t "gai" -n "dql"   "python main.py --algo dql  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window   -t "gai" -n "ppo"   "python main.py --algo dppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window   -t "gai" -n "gppo"  "python main.py --algo gppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window   -t "gai" -n "gdql"  "python main.py --algo gdql --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window   -t "gai" -n "a2c"   "python main.py --algo a2c  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 
tmux new-window   -t "gai" -n "da2c"  "python main.py --algo da2c --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME" 