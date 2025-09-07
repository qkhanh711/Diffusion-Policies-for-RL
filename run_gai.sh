#!/bin/bash2
SEED=24
EPISODES=1500
SAVE_EPISODES=300                                                                                                                                                                                                                                                               
ENV_NAME="gail-service-env-v6-baseline"
DIR="parallel_results"
TMUX_NAME="t_gai"
FILE_MAIN="src/parallel_main.py"
DEVICE=1
PRINT=False
NUM_USERS=10

# rm -r parallel_results/
# tmux kill-session -t "$TMUX_NAME"
# tmux new-session -d -s "$TMUX_NAME" -n "watch" "watch -n 1 nvidia-smi"

tmux new-window   -t "$TMUX_NAME" -n "ppo"  "python $FILE_MAIN --algo dppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS --device $DEVICE"
tmux new-window   -t "$TMUX_NAME" -n "gppo" "python $FILE_MAIN --algo gppo --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS"
tmux new-window   -t "$TMUX_NAME" -n "dql"  "python $FILE_MAIN --algo dql  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS"
tmux new-window   -t "$TMUX_NAME" -n "gdql" "python $FILE_MAIN --algo gdql --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS --device $DEVICE"
# tmux new-window   -t "$TMUX_NAME" -n "a2c"  "python $FILE_MAIN --algo a2c  --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS"
# tmux new-window   -t "$TMUX_NAME" -n "da2c" "python $FILE_MAIN --algo da2c --seed $SEED --num_episodes $EPISODES --env_name $ENV_NAME --dir $DIR --save_episodes $SAVE_EPISODES --n_users $NUM_USERS --device $DEVICE"

tmux detach -s "$TMUX_NAME"
