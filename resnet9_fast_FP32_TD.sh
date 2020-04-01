block_size=${1:-4}
gamma=${2:-0.0}
alpha=${3:-0.0}
gamma_final=${4:-0.875}
alpha_final=${5:-0.99}
ramping_power=${6:-5.0}


python main.py --block_size $block_size \
            --TD_gamma $gamma \
            --TD_alpha $alpha \
            --TD_gamma_final $gamma_final \
            --TD_alpha_final $alpha_final \
            --ramping_power $ramping_power \
