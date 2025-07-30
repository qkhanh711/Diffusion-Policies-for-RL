#  gaurantee that the session is running
# clear session and re run
tmux kill-session -t "rl"

# tmux new-session -s "rl" -n "ql" "python UAVtest.py --algo ql"
# tmux new-session -s "rl" -n "gdql" "python UAVtest.py --algo gdql"
# tmux new-window -t "rl" -n "ppo" "python UAVtest.py --algo ppo"
# tmux new-window -t "rl" -n "gppo" "python UAVtest.py --algo gppo"
# tmux new-window -t "rl" -n "a2c" "python UAVtest.py --algo a2c"
# tmux new-window -t "rl" -n "bc" "python UAVtest.py --algo bc"
# tmux new-window -t "rl" -n "dql" "python UAVtest.py --algo dql"

tmux new-session -s "rl" -n "gdql" "python UAV_test_new.py --algo gdql"
tmux new-window -t "rl" -n "ppo" "python UAV_test_new.py --algo ppo"
tmux new-window -t "rl" -n "gppo" "python UAV_test_new.py --algo gppo"
tmux new-window -t "rl" -n "a2c" "python UAV_test_new.py --algo a2c"
# tmux new-window -t "rl" -n "bc" "python UAV_test_new.py --algo bc"
tmux new-window -t "rl" -n "dql" "python UAV_test_new.py --algo dql"