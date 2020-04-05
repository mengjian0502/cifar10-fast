block_size=${1:-4}
gamma=${2:-0.0}
alpha=${3:-0.0}
gamma_final=${4:-0.9375}
alpha_final=${5:-0.99}
ramping_power=${6:-5.0}

save_file="resnet9_FP32_TD_${block_size}_${gamma_final}_${alpha_final}_ramping_power_${ramping_power}.pth"

python main.py --block_size $block_size \
            --TD_gamma $gamma \
            --TD_alpha $alpha \
            --TD_gamma_final $gamma_final \
            --TD_alpha_final $alpha_final \
            --ramping_power $ramping_power \
            --save_file $save_file \
            --epochs 35 \
