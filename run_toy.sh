tmux kill-session -t "rl2"

tmux new-session -s "rl2" -n "ql" "python main.py --algo ql"
tmux new-session -s "rl2" -n "gdql" "python main.py --algo gdql"
tmux new-window -t "rl2" -n "gppo" "python main.py --algo gppo"
