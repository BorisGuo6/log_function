#!/bin/bash
# for name in 27m_vs_30m 3s5z_vs_3s6z 6h_vs_8z corridor MMM2
for name in CartPole_avgreturn CartPole_loss CartPole_ratio
do
    python draw_benchmark.py \
    --game $name
done
