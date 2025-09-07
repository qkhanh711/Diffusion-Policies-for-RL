#!/bin/bash
SEED=1
EPISODES=1200
ENV_NAME="gail-service-env-v3-org"
DIR="t_results"
TMUX_NAME="t_gai"
FILE_MAIN="src/t_main.py"
DEVICE=1
PRINT=False
NUM_USERS=12

# rm -r 
tmux kill-session -t "$TMUX_NAME"
tmux new-session -d -s "$TMUX_NAME" -n "watch" "watch -n 1 nvidia-smi"

tmux new-window   -t "$TMUX_NAME" -n "ppo"  "python $FILE_MAIN --algo dppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --n_users $NUM_USERS --device $DEVICE"
tmux new-window   -t "$TMUX_NAME" -n "gppo" "python $FILE_MAIN --algo gppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --n_users $NUM_USERS"
tmux new-window   -t "$TMUX_NAME" -n "dql"  "python $FILE_MAIN --algo dql  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --n_users $NUM_USERS"
tmux new-window   -t "$TMUX_NAME" -n "gdql" "python $FILE_MAIN --algo gdql --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --n_users $NUM_USERS --device $DEVICE"
# tmux new-window   -t "$TMUX_NAME" -n "a2c"  "python $FILE_MAIN --algo a2c  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR"
# tmux new-window   -t "$TMUX_NAME" -n "da2c" "python $FILE_MAIN --algo da2c --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR"

tmux detach -s "$TMUX_NAME"
