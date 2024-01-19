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
        if alg == 'actor_critic_loge':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log(pi_i.probs.gather(0, act[i].to(device)))
                # print('logp_a_i =', logp_a_i)
                logp_a.append(logp_a_i)
            return logp_a
        elif alg == 'actor_critic_log2':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log2(pi_i.probs.gather(0, act[i].to(device)))
                logp_a.append(logp_a_i)
            return logp_a
        elif alg == 'actor_critic_log10':
            for i, pi_i in enumerate(pi):
                logp_a_i = torch.log10(pi_i.probs.gather(0, act[i].to(device)))
                logp_a.append(logp_a_i)
            return logp_a
        else:
            return pi.log_prob(act)


# !暂时没用
class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation).to(device)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
        print('-------------------------------------------------------------------')
        print('pi =', pi)
        print('act =', act)
        print('pi.log_prob(act).sum(axis=-1) =', pi.log_prob(act).sum(axis=-1))
        if self.alg == 'actor_critic_loge':
            print('loge_prob_ours(pi, act).sum(axis=-1) =', loge_prob_ours(pi, act).sum(axis=-1))
            return loge_prob_gaussian_ours(pi, act).sum(axis=-1)
        elif self.alg == 'actor_critic_log2':
            print('log2_prob_ours(pi, act).sum(axis=-1) =', log2_prob_ours(pi, act).sum(axis=-1))
            return log2_prob_gaussian_ours(pi, act).sum(axis=-1)
        elif self.alg == 'actor_critic_log10':
            print('log10_prob_ours(pi, act).sum(axis=-1) =', log10_prob_ours(pi, act).sum(axis=-1))
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
        return pi, a.detach().cpu().numpy(), v

    def log_prob_from_distribution(self, alg, pi, a):
        logp_a = self.pi._log_prob_from_distribution(alg, pi, a)
        return logp_a

    def act(self, obs):
        return self.step(obs)[0]


def train(env, ac, pi_optimizer, vf_optimizer, discount_factor):
    alg = 'actor_critic_loge'
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

    # 根据方差选择log
    # print('--------------------------------------------------------------------------')
    # print('returns =', returns)
    returns_mean = torch.mean(returns)
    returns_var = torch.var(returns)
    returns_std = torch.std(returns)
    # print('returns_var =', returns_var)
    # print('mean : {} | std : {} | std/mean : {}'.format(returns_mean, returns_std, returns_std/returns_mean))

    returns_pos = []
    returns_neg = []
    for _, item_i in enumerate(returns):
        if item_i >= 0:
            returns_pos.append(item_i)
        else:
            returns_neg.append(item_i)

    returns_pos_mean = np.mean(returns_pos)
    returns_neg_mean = np.mean(returns_neg)
    returns_pos_std = np.std(returns_pos)
    returns_neg_std = np.std(returns_neg)
    returns_pos_ratio = np.true_divide(returns_pos_std, returns_pos_mean)
    returns_neg_ratio = np.true_divide(returns_neg_std, abs(returns_neg_mean))
    returns_ratio = np.true_divide(returns_pos_ratio+returns_neg_ratio, 2)
    # print('returns_ratio =', returns_ratio)
    
    # returns_ratio = np.where(returns_mean != 0, np.true_divide(returns_std, returns_mean), np.nan)
    tb_writer.add_scalar('std_absmean_ratio', returns_ratio, episode)

    # print('pos mean : {} | std : {} | std/mean : {}'.format(returns_pos_mean, returns_pos_std, returns_pos_ratio))
    # print('neg mean : {} | std : {} | std/abs(mean) : {}'.format(returns_neg_mean, returns_neg_std, returns_neg_ratio))
    
    # if returns_ratio < 0.1:
    #     alg == 'actor_critic_log2'
    # elif 0.3 <= returns_ratio < 0.57:
    #     alg == 'actor_critic_loge'
    # else:
    #     alg == 'actor_critic_log10'
    
    # alg = 'actor_critic_loge'
    # alg = 'actor_critic_log2'
    alg = 'actor_critic_log10'
    
    log_prob_actions = ac.log_prob_from_distribution(alg, pis, actions)
    log_prob_actions = torch.stack(log_prob_actions)

    policy_loss, value_loss = update_policy(returns, log_prob_actions, values, pi_optimizer, vf_optimizer)

    #! 根据方差选择是否降低loss
    large_variance_threshold = 5  # This is an arbitrary threshold; adjust as needed
    if returns_var > large_variance_threshold:
        policy_loss *= 0.1
        value_loss *= 0.1
    
    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize = True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r +  R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def update_policy(returns, log_prob_actions, values, pi_optimizer, vf_optimizer):
    returns = returns.detach().to(device)
    log_prob_actions = log_prob_actions.to(device)
    policy_loss = - (returns * log_prob_actions).sum()
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
            _, action_pred, _, = ac.step(torch.as_tensor(state, dtype=torch.float32).to(device))

        state, reward, done, _ = env.step(action_pred)
        episode_reward += reward

    return episode_reward


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Actor-Critic (AC) agent')
    # env
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help="CartPole-v1, LunarLander-v2, Acrobot-v1")
    parser.add_argument('--render', action='store_true', default=False, help="render")
    # RL
    parser.add_argument('--pi_lr', type=float, default=3e-4, help="pi learning rate: 1e-2, 1e-3")
    parser.add_argument('--vf_lr', type=float, default=1e-3, help="vf learning rate: 1e-2, 1e-3")
    parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor")
    # traing
    parser.add_argument('--optimizer', type=str, default='Adam', help="torch optimizer: SGD, Adam")
    parser.add_argument('--hidden_dim', type=int, default=256, help="number of hidden dim of network")
    parser.add_argument('--l', type=int, default=2)
    # seed,换用3个不同的seed
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
        train_env = gym.make("LunarLander-v2", continuous=True)
        test_env = gym.make("LunarLander-v2", continuous=True)
    # elif args.env_name == 'CartPole-v1' :
    #     train_env = gym.make('CartPole-v1', new_step_api=True)
    #     test_env = gym.make('CartPole-v1', new_step_api=True)
    else:
        train_env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)

    obs_dim = train_env.observation_space.shape
    act_dim = train_env.action_space.shape

    ac_kwargs = dict(hidden_sizes=[args.hidden_dim]*args.l)
    # Create actor-critic module
    ac = MLPActorCritic(train_env.observation_space, train_env.action_space, **ac_kwargs).to(device)
    # ac = MLPGaussianActor(train_env.observation_space, train_env.action_space, **ac_kwargs).to(device)
    
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

    train_rewards = []
    test_rewards = []

    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    log_name = "alg-actor_critic_logad_env-{}_optimizer-{}_pi_lr-{}_seed-{}_{}".format(args.env_name, args.optimizer, args.pi_lr, args.seed, suffix)
    log_dir = os.path.join('/home2/ad/liuqi/log_fn_log10/tensorboard_log/' + log_name)
    print('log_dir =', log_dir)
    tb_writer = SummaryWriter(log_dir)

    for episode in range(1, MAX_EPISODES+1):
        policy_loss, critic_loss, train_reward = train(train_env, ac, pi_optimizer, vf_optimizer, DISCOUNT_FACTOR)
        tb_writer.add_scalar('policy_loss', policy_loss, episode)
        tb_writer.add_scalar('critic_loss', critic_loss, episode)

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
