import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import gym
from gym.spaces import Box, Discrete

import math
import numpy as np
from numbers import Real

import os
import argparse
import datetime
from tensorboardX import SummaryWriter


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def loge_prob_gaussian_ours(pi, value):
    # compute the variance
    var = (pi.scale ** 2)
    log_scale = math.loge(pi.scale) if isinstance(pi.scale, Real) else pi.scale.log()
    return -((value - pi.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


def log2_prob_gaussian_ours(pi, value):
    var = (pi.scale ** 2)
    log_scale = math.log2(pi.scale) if isinstance(pi.scale, Real) else pi.scale.log2()
    return -((value - pi.loc) ** 2) / (2 * var) - log_scale - math.log2(math.sqrt(2 * math.pi))


def log10_prob_gaussian_ours(pi, value):
    var = (pi.scale ** 2)
    log_scale = math.log10(pi.scale) if isinstance(pi.scale, Real) else pi.scale.log10()
    return -((value - pi.loc) ** 2) / (2 * var) - log_scale - math.log10(math.sqrt(2 * math.pi))


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, alg, pi, act):
        logp_a = []
        if alg == 'gae_loge':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log(pi_i.probs.gather(0, act[i].to(device)))
                # print('logp_a_i =', logp_a_i)
                logp_a.append(logp_a_i)
            return logp_a
        elif alg == 'gae_log2':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log2(pi_i.probs.gather(0, act[i].to(device)))
                logp_a.append(logp_a_i)
            return logp_a
        elif alg == 'gae_log10':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log10(pi_i.probs.gather(0, act[i].to(device)))
                logp_a.append(logp_a_i)
            return logp_a
        else:
            return pi.log_prob(act)


# !暂时没用
class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, alg, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation).to(device)
        self.alg = alg

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
        # print('-------------------------------------------------------------------')
        # print('pi =', pi)
        # print('act =', act)
        # print('pi.log_prob(act).sum(axis=-1) =', pi.log_prob(act).sum(axis=-1))
        if self.alg == 'gae_loge':
            # print('loge_prob_ours(pi, act).sum(axis=-1) =', loge_prob_ours(pi, act).sum(axis=-1))
            return loge_prob_gaussian_ours(pi, act).sum(axis=-1)
        elif self.alg == 'gae_log2':
            # print('log2_prob_ours(pi, act).sum(axis=-1) =', log2_prob_ours(pi, act).sum(axis=-1))
            return log2_prob_gaussian_ours(pi, act).sum(axis=-1)
        elif self.alg == 'gae_log10':
            # print('log10_prob_ours(pi, act).sum(axis=-1) =', log10_prob_ours(pi, act).sum(axis=-1))
            return log10_prob_gaussian_ours(pi, act).sum(axis=-1)
        else:
            return pi.log_prob(act).sum(axis=-1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation).to(device)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation).to(device)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation).to(device)

    def step(self, obs):
        # with torch.no_grad():
        pi = self.pi._distribution(obs)
        a = pi.sample()
        v = self.v(obs)
        # print('a =', a)
        # print('v =', v)
        # print('logp_a =', logp_a)
        return pi, a.detach().cpu().numpy(), v

    def log_prob_from_distribution(self, alg, pi, a):
        logp_a = self.pi._log_prob_from_distribution(alg, pi, a)
        return logp_a

    def act(self, obs):
        return self.step(obs)[0]


def train(env, ac, pi_optimizer, vf_optimizer, discount_factor, trace_decay):
    alg = 'gae_loge'
    pis = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        pi, action_pred, value_pred = ac.step(torch.as_tensor(state, dtype=torch.float32).to(device))

        # state, reward, done, _ = env.step(action.item())
        state, reward, done, _ = env.step(action_pred)

        pis.append(pi)
        actions.append(torch.as_tensor(action_pred, dtype=torch.int64))
        values.append(torch.as_tensor(value_pred, dtype=torch.float32).unsqueeze(0))
        rewards.append(torch.as_tensor(reward, dtype=torch.float32).unsqueeze(0))

        episode_reward += reward

    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    #note: calculate_advantages takes in rewards, not returns!
    advantages = calculate_advantages(rewards, values, discount_factor, trace_decay)

    # 根据方差选择log
    # print('--------------------------------------------------------------------------')
    # print('advantages =', advantages)
    # advantages_mean = torch.mean(advantages)
    # advantages_var = torch.var(advantages)
    # advantages_std = torch.std(advantages)
    # print('mean : {} | std : {} | std/mean : {}'.format(advantages_mean, advantages_std, advantages_std/advantages_mean))
    with torch.no_grad():
        advantages_pos = []
        advantages_neg = []
        for _, item_i in enumerate(advantages):
            if item_i >= 0:
                advantages_pos.append(item_i)
            else:
                advantages_neg.append(item_i)

        # print('advantages_pos =', advantages_pos)
        # advantages_pos_mean = np.mean(advantages_pos)
        # advantages_neg_mean = np.mean(advantages_neg)
        # advantages_pos_std = np.std(advantages_pos)
        # advantages_neg_std = np.std(advantages_neg)
        # advantages_pos_ratio = np.true_divide(advantages_pos_std, advantages_pos_mean)
        # advantages_neg_ratio = np.true_divide(advantages_neg_std, abs(advantages_neg_mean))
        # advantages_ratio = np.true_divide(advantages_pos_ratio+advantages_neg_ratio, 2)

        advantages_pos_mean = torch.mean(torch.stack(advantages_pos))
        advantages_neg_mean = torch.mean(torch.stack(advantages_neg))
        advantages_pos_std = torch.std(torch.stack(advantages_pos))
        advantages_neg_std = torch.std(torch.stack(advantages_neg))
        advantages_pos_ratio = torch.true_divide(advantages_pos_std, advantages_pos_mean)
        advantages_neg_ratio = torch.true_divide(advantages_neg_std, abs(advantages_neg_mean))
        advantages_ratio = torch.true_divide(advantages_pos_ratio+advantages_neg_ratio, 2)
        tb_writer.add_scalar('advantages_std_absmean_ratio', advantages_ratio, episode)

    # print('pos mean : {} | std : {} | std/mean : {}'.format(advantages_pos_mean, advantages_pos_std, advantages_pos_ratio))
    # print('neg mean : {} | std : {} | std/abs(mean) : {}'.format(advantages_neg_mean, advantages_neg_std, advantages_neg_ratio))
    # print('advantages_ratio =', advantages_ratio)
    if advantages_ratio < 0.3:
        alg == 'actor_critic_log2'
    elif 0.2 < advantages_ratio < 0.6:
        alg == 'actor_critic_loge'
    else:
        alg == 'actor_critic_log10'
    log_prob_actions = ac.log_prob_from_distribution(alg, pis, actions)
    log_prob_actions = torch.stack(log_prob_actions)

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, pi_optimizer, vf_optimizer)

    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize = True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns).to(device)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def calculate_advantages(rewards, values, discount_factor, trace_decay, normalize = True):
    advantages = []
    advantage = 0
    next_value = 0
    next_value = torch.tensor(next_value).to(device)

    for r, v in zip(reversed(rewards), reversed(values)):
        r = torch.tensor(r).to(device)
        v = torch.tensor(v).to(device)
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * trace_decay
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages).to(device)

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def update_policy(advantages, log_prob_actions, returns, values, pi_optimizer, vf_optimizer):
    advantages = advantages.detach().to(device)
    log_prob_actions = log_prob_actions.to(device)
    returns = returns.detach().to(device)

    policy_loss = - (advantages * log_prob_actions).sum()
    # value_loss = F.smooth_l1_loss(returns, values).sum()
    value_loss = F.smooth_l1_loss(returns.float(), values).sum()

    pi_optimizer.zero_grad()
    vf_optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    pi_optimizer.step()
    vf_optimizer.step()

    return policy_loss.item(), value_loss.item()


def evaluate(env, ac):
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        with torch.no_grad():
            _, action_pred, _ = ac.step(torch.as_tensor(state, dtype=torch.float32).to(device))

        state, reward, done, _ = env.step(action_pred)
        episode_reward += reward

    return episode_reward


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generalized_Advantage_Estimation (GAE) agent')
    # env
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help="CartPole-v1, LunarLander-v2, Acrobot-v1")
    parser.add_argument('--render', action='store_true', default=False, help="render")
    # RL
    parser.add_argument('--pi_lr', type=float, default=3e-4, help="pi learning rate: 1e-2, 1e-3")
    parser.add_argument('--vf_lr', type=float, default=1e-3, help="vf learning rate: 1e-2, 1e-3")
    parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor")
    parser.add_argument('--trace_decay', type=float, default=0.99, help="trace_decay")
    # traing
    parser.add_argument('--optimizer', type=str, default='SGD', help="torch optimizer: SGD, Adam")
    parser.add_argument('--hidden_dim', type=int, default=128, help="number of hidden dim of network")
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--num_episodes', type=int, default=1000, help="num_episodes")
    # print
    parser.add_argument('--n_trail', type=int, default=25, help="n_trail")
    parser.add_argument('--print_every', type=int, default=10, help="print_every")
    args = parser.parse_args()

    if args.env_name == 'BipedalWalker-v3_hardcore-False':
        train_env = gym.make("BipedalWalker-v3", hardcore=False)
        test_env = gym.make("BipedalWalker-v3", hardcore=False)
    elif args.env_name == 'LunarLander-v2_continuous-True':
        train_env = gym.make("LunarLander-v2", continuous= True)
        test_env = gym.make("LunarLander-v2", continuous= True)
    else:
        train_env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)

    obs_dim = train_env.observation_space.shape
    act_dim = train_env.action_space.shape

    ac_kwargs = dict(hidden_sizes=[args.hidden_dim]*args.l)
    # Create actor-critic module
    ac = MLPActorCritic(train_env.observation_space, train_env.action_space, **ac_kwargs).to(device)

    PI_LR = args.pi_lr
    VF_LR = args.vf_lr
    if args.optimizer == 'SGD':
        pi_optimizer = torch.optim.SGD(ac.pi.parameters(), lr=PI_LR, momentum=0)
        vf_optimizer = torch.optim.SGD(ac.v.parameters(), lr=VF_LR, momentum=0)
    else:
        pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=PI_LR)
        vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=VF_LR)

    SEED = args.seed
    train_env.seed(SEED)
    test_env.seed(SEED+1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    MAX_EPISODES = args.num_episodes
    DISCOUNT_FACTOR = args.discount_factor
    N_TRIALS = args.n_trail
    PRINT_EVERY = args.print_every
    TRACE_DECAY = args.trace_decay

    train_rewards = []
    test_rewards = []

    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    log_name = "alg-gae_logad_env-{}_optimizer-{}_pi_lr-{}_seed-{}_{}".format(args.env_name, args.optimizer, args.pi_lr, args.seed, suffix)
    log_dir = os.path.join('./tensorboard_log/' + log_name)
    print('log_dir =', log_dir)
    tb_writer = SummaryWriter(log_dir)

    for episode in range(1, MAX_EPISODES+1):
        policy_loss, value_loss, train_reward = train(train_env, ac, pi_optimizer, vf_optimizer, DISCOUNT_FACTOR, TRACE_DECAY)
        tb_writer.add_scalar('policy_loss', policy_loss, episode)
        tb_writer.add_scalar('critic_loss', value_loss, episode)

        test_reward = evaluate(test_env, ac)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        tb_writer.add_scalar('original_train_returns', train_rewards[-1], episode)
        tb_writer.add_scalar('original_test_returns', test_rewards[-1], episode)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        tb_writer.add_scalar('mean_train_returns', mean_train_rewards, episode)
        tb_writer.add_scalar('mean_test_returns', mean_test_rewards, episode)

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')