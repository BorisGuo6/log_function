#!/bin/bash

# optimizer: SGD, Adam

# CartPole-v1 LunarLander-v2 Acrobot-v1
# BipedalWalker-v3 Pendulum-v1 MountainCarContinuous-v0
# Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
# for env_name in HalfCheetah-v4 Ant-v4 Hopper-v4 Humanoid-v4
for env_name in MountainCarContinuous-v0
# for env_name in Acrobot-v1 
# for env_name in Pendulum-v1
# for env_name in BipedalWalker-v3 
# for env_name in LunarLander-v2
# for env_name in CartPole-v1
do
    for optimizer in SGD
    do
        for num_episodes in 3000
        do
            for pi_lr in 0.0003
            do
                for seed in 0 1 2
                do
                    python /home2/ad/liuqi/log_fn_adaptive/actor_critic.py \
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


# for alg in actor_critic_loge
# do
#     # for env_name in CliffWalking-v0 Taxi-v3 FrozenLake-v1 lackjack-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in HalfCheetah-v4 Ant-v4 Hopper-v4 Humanoid-v4
#     # for env_name in HalfCheetah-v3 Ant-v3 Hopper-v3 Humanoid-v3
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in BipedalWalker-v3 Pendulum-v1
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
#                         python actor_critic.py \
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


# for alg in actor_critic actor_critic_loge actor_critic_log2 actor_critic_log10
# do
#     # for env_name in CliffWalking-v0 Taxi-v3 FrozenLake-v1 lackjack-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in HalfCheetah-v4 Ant-v4 Hopper-v4 Humanoid-v4
#     # for env_name in HalfCheetah-v3 Ant-v3 Hopper-v3 Humanoid-v3
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in BipedalWalker-v3 Pendulum-v1
#     for env_name in BipedalWalker-v3_hardcore-False LunarLander-v2_continuous-True
#     do
#         for optimizer in SGD Adam
#         do
#             for num_episodes in 3000
#             do
#                 for seed in 0 1 2
#                 do
#                     python actor_critic.py \
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


# # for alg in actor_critic actor_critic_loge actor_critic_log2 actor_critic_log10
# for alg in actor_critic_loge
# do
#     # for env_name in CartPole-v1 LunarLander-v2 Acrobot-v1
#     # for env_name in CliffWalking-v0 Taxi-v3 FrozenLake-v1 lackjack-v1
#     # for env_name in Catcher-PLE-v0 Pixelcopter-PLE-v0 Pong-PLE-v0
#     # for env_name in HalfCheetah-v4 Ant-v4 Hopper-v4 Humanoid-v4
#     # for env_name in HalfCheetah-v3 Ant-v3 Hopper-v3 Humanoid-v3
#     for env_name in MountainCarContinuous-v0
#     do
#         for optimizer in SGD
#         do
#             for num_episodes in 2000
#             do
#                 for seed in 0
#                 do
#                     python actor_critic.py \
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