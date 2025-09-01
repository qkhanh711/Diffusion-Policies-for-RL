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

# tmux new-session -s "rl" -n "ql" "python test.py --algo ql"
tmux new-session -s "mrl" -n "gdql" "python test.py --algo gdql"
# tmux new-window -t "mrl" -n "ppo" "python test.py --algo ppo"
tmux new-window -t "mrl" -n "gppo" "python test.py --algo gppo"
# tmux new-window -t "mrl" -n "a2c" "python test.py --algo a2c"
# tmux new-window -t "mrl" -n "bc" "python test.py --algo bc"
# tmux new-window -t "mrl" -n "dql" "python test.py --algo dql"