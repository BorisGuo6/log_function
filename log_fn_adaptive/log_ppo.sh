#!/bin/bash

# CartPole-v1 LunarLander-v2 Acrobot-v1
# Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
# optimizer: SGD, Adam


for alg in ppo_loge ppo_log2 ppo_log10 ppo
do
    # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
    for env_name in CliffWalking-v0 Taxi-v3 FrozenLake-v1 lackjack-v1
    # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
    do
        for optimizer in SGD Adam
        do
            # for lr in 0.01 0.001
            for lr in 0.001
            do
                for num_episodes in 1000
                do
                    for seed in 0 1 2
                    do
                        python ppo.py \
                        --alg $alg \
                        --env_name $env_name \
                        --optimizer $optimizer \
                        --lr $lr \
                        --seed $seed \
                        --num_episodes $num_episodes
                    done
                done
            done
        done
    done
done