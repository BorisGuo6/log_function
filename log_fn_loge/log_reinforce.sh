#!/bin/bash

# CartPole-v1 LunarLander-v2 Acrobot-v1
# Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
# CliffWalking-v0 Taxi-v3 FrozenLake-v1 lackjack-v1
# BipedalWalker-v3 Pendulum-v1
# optimizer: SGD, Adam


# for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
# for env_name in CartPole-v1
# for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
# for env_name in BipedalWalker-v3_hardcore-False LunarLander-v2_continuous-True
for env_name in CartPole-v1 LunarLander-v2
do
    for optimizer in SGD
    do
        for num_episodes in 3000
        do
            for pi_lr in 0.0003
            do
                for seed in 0 1 2
                do
                    python REINFORCE.py \
                    --env_name $env_name \
                    --optimizer $optimizer \
                    --seed $seed \
                    --pi_lr $pi_lr \
                    --num_episodes $num_episodes
                done
            done
        done
    done
done


# for alg in reinforce_loge
# do
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in CartPole-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in BipedalWalker-v3_hardcore-False LunarLander-v2_continuous-True
#     for env_name in CartPole-v1 LunarLander-v2
#     do
#         for optimizer in SGD
#         do
#             for num_episodes in 3000
#             do
#                 for pi_lr in 0.003 0.00003
#                 do
#                     for seed in 0 1 2
#                     do
#                         python REINFORCE.py \
#                         --alg $alg \
#                         --env_name $env_name \
#                         --optimizer $optimizer \
#                         --seed $seed \
#                         --pi_lr $pi_lr \
#                         --num_episodes $num_episodes
#                     done
#                 done
#             done
#         done
#     done
# done


# for alg in reinforce reinforce_loge reinforce_log2 reinforce_log10
# do
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in CartPole-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in BipedalWalker-v3_hardcore-False LunarLander-v2_continuous-True
#     for env_name in CartPole-v1 LunarLander-v2
#     do
#         for optimizer in SGD
#         do
#             for num_episodes in 3000
#             do
#                 for seed in 0 1 2
#                 do
#                     python REINFORCE.py \
#                     --alg $alg \
#                     --env_name $env_name \
#                     --optimizer $optimizer \
#                     --seed $seed \
#                     --num_episodes $num_episodes
#                 done
#             done
#         done
#     done
# done


# for alg in reinforce
# do
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in CartPole-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in BipedalWalker-v3_hardcore-False LunarLander-v2_continuous-True
#     for env_name in LunarLander-v2_continuous-True
#     do
#         for optimizer in SGD
#         do
#             for num_episodes in 3000
#             do
#                 for seed in 0
#                 do
#                     python REINFORCE.py \
#                     --alg $alg \
#                     --env_name $env_name \
#                     --optimizer $optimizer \
#                     --seed $seed \
#                     --num_episodes $num_episodes
#                 done
#             done
#         done
#     done
# done