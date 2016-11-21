killall -v tensorboard
rm -rf tensorboard/*
clear
ipython -i ${1:-run_rnn_with_artificial_sequential_data.py}
