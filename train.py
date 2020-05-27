# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from data import get_recent_data
from model import *
from data import load_all_data
from utils import *
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from optimize import get_wealths
import math
from parameters import commissions, spread, spread_bear, spread_bull

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def get_aggregated_and_multiplied_X(X, aggregate_N, m, m_bear = None):
    short = m_bear is not None
    X_orig = X

    if short:
        X_bear = get_multiplied_X(X, -m_bear)
    if m > 1:
        X = get_multiplied_X(X, m)

    X_agg = aggregate(X, aggregate_N)
    X_orig_agg = aggregate(X_orig, aggregate_N)
    if short:
        X_bear_agg = aggregate(X_bear, aggregate_N)

    if short:
        return X_orig_agg, X_agg, X_bear_agg

    return X_orig_agg, X_agg

# TODO: better variable names?
def get_rewards(X, c, d, X_bear, m):
    short = X_bear is not None
    if short:
        assert(X.shape == X_bear.shape)
    idx = torch.arange(d)
    weights = torch.exp(- c * idx.float())
    weights /= weights.sum()

    n = X.shape[0] - d

    returns = torch.zeros(n, d)
    if short:
        returns_bear = torch.zeros(n, d)

    for i in range(1, d + 1):
        returns[:, i - 1] = torch.from_numpy(X[i:i+n, 0] / X[:n, 0])
        if short:
            returns_bear[:, i - 1] = torch.from_numpy(X_bear[i:i+n, 0] / X_bear[:n, 0])

    keep_rewards = torch.log(torch.matmul(returns, weights)).view(-1, 1)
    if short:
        do_nothing_rewards = torch.log(torch.matmul(returns_bear, weights)).view(-1, 1)
    else:
        do_nothing_rewards = torch.zeros(n, 1)

    log_commissions_and_spread_bull = torch.log(torch.ones(1) - commissions - (spread_bull if m > 1 else spread))

    buy_rewards = keep_rewards + log_commissions_and_spread_bull
    sell_rewards = do_nothing_rewards + log_commissions_and_spread_bull
    if short:
        log_commissions_and_spread_bear = torch.log(torch.ones(1) - commissions - spread_bear)
        buy_rewards += log_commissions_and_spread_bear
        sell_rewards += log_commissions_and_spread_bear

    keep_rewards = torch.cat([sell_rewards, keep_rewards], dim = 1)
    buy_rewards = torch.cat([do_nothing_rewards, buy_rewards], dim = 1)

    return keep_rewards, buy_rewards

def get_optim_decisions(keep_rewards, buy_rewards):
    keeps = torch.argmax(keep_rewards, dim = 1).float()
    buys = torch.argmax(buy_rewards, dim = 1).float()

    return keeps, buys

# TODO: make a version that doesn't make this assumption
# assumes keeps and buys are 1-D binary vectors
def get_realized_decisions(keeps, buys):
    li = keeps == buys
    out = torch.zeros(keeps.shape[0])
    out[li] = keeps[li]

    idx = torch.arange(keeps.shape[0])

    # li = points that need to be processed
    li = ~li

    if li[0]:
        out[0] = buys[0]
        li[0] = False

    while li.sum() > 0:
        idx1 = idx[li]
        idx_m = idx1[torch.cat([torch.ones(1).bool(), idx1[1:] - idx1[:-1] > 1])]

        out_prev = out[idx_m - 1]

        idx_1 = idx_m[out_prev == 1]
        idx_0 = idx_m[out_prev == 0]

        out[idx_1] = keeps[idx_1]
        out[idx_0] = buys[idx_0]

        li[idx_m] = False

    return out

# assumes keeps and buys are 1-D binary vectors
def get_realized_decisions_old(keeps, buys):
    li = keeps == buys
    out = torch.zeros(keeps.shape[0])
    out[li] = keeps[li]

    idx = torch.arange(keeps.shape[0])
    not_same_idx = idx[~li]

    for i in not_same_idx:
        if i > 0 and out[i - 1] == 1:
            out[i] = keeps[i]
        else:
            out[i] = buys[i]

    return out

def optimize_reward_function(X, n_runs, kappa, aggregate_N, X_bear = None, m = 1):
    def objective_function(c, d):
        c = 10 ** c
        d = int(d)

        keep_rewards, buy_rewards = get_rewards(X, c, d, X_bear, m)

        keeps, buys = get_optim_decisions(keep_rewards, buy_rewards)

        buys = get_realized_decisions(keeps, buys).numpy()

        N = buys.shape[0]

        wealths = get_wealths(
            X[:N, :],
            buys,
            commissions = commissions,
            spread_bull = spread if m <= 1.0 else spread_bull,
            X_bear = X_bear[:N, :],
            spread_bear = spread_bear
        )

        n_months = N * aggregate_N / (24 * 30)

        wealth = wealths[-1] ** (1 / n_months)

        print('c = {}, d = {}, log wealth = {}'.format(
            round_to_n(c, 3),
            d,
            round_to_n(np.log(wealth), 4)
        ))

        return np.log(wealth)

    # Bounded region of parameter space
    pbounds = {
        'c': (-4, 1),
        'd': (1.0, 60.0)
    }

    optimizer = BayesianOptimization(
        f = objective_function,
        pbounds = pbounds,
        # random_state = 1,
    )

    filename = 'optim_results/reward.json'

    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    optimizer.probe(
        params={
            'c': 1,
            'd': 60,
        },
        lazy=True,
    )

    optimizer.maximize(
        init_points = int(np.sqrt(n_runs)),
        n_iter = n_runs,
        kappa = kappa,
    )

    print(optimizer.max)

def plot_c_and_d(X, X_orig, c, d, aggregate_N, X_bear = None, m = 1):
    keep_rewards, buy_rewards = get_rewards(X, c, d, X_bear, m)

    keeps, buys = get_optim_decisions(keep_rewards, buy_rewards)

    buys = get_realized_decisions(keeps, buys).numpy()

    N = buys.shape[0]

    wealths = get_wealths(
        X[:N, :],
        buys,
        commissions = commissions,
        spread_bull = spread if m <= 1.0 else spread_bull,
        X_bear = X_bear[:N, :],
        spread_bear = spread_bear
    )

    n_months = N * aggregate_N / (24 * 30)

    wealth = wealths[-1] ** (1 / n_months)

    print('n_months = {}, wealth = {}'.format(
        round_to_n(n_months, 3),
        round_to_n(wealth, 4)
    ))

    plt.style.use('seaborn')
    plt.plot(X_orig[:N, 0] / X_orig[0, 0], c='k', alpha=0.5)
    plt.plot(wealths, c='g')
    if np.log(wealths[-1]) / np.log(10) > 2.5:
        plt.yscale('log')
    plt.show()

def get_reward(keeps, buys, keep_rewards, buy_rewards):
    keeps = keeps.view(-1, 1)
    keeps = torch.cat([1 - keeps, keeps], dim = 1)

    buys = buys.view(-1, 1)
    buys = torch.cat([1 - buys, buys], dim = 1)

    keep_reward = torch.sum(keep_rewards * keeps)
    buy_reward = torch.sum(buy_rewards * buys)

    return keep_reward + buy_reward

def train(
    X,
    aggregate_N,
    m,
    m_bear,
    model,
    lr,
    n_epochs,
    print_step,
    c,
    d,
):
    plt.style.use('seaborn')
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    X_orig, X, X_bear = get_aggregated_and_multiplied_X(X, aggregate_N, m, m_bear)

    keep_rewards, buy_rewards = get_rewards(X, c, d, X_bear, m)

    sequence_length = model.sequence_length
    keep_rewards = keep_rewards[sequence_length - 1:, :]
    buy_rewards = buy_rewards[sequence_length - 1:, :]

    N = keep_rewards.shape[0]

    weatlhs_line, = ax.plot(np.arange(N), np.arange(N), 'g')
    ax.plot(X_orig[-N:, 0] / X_orig[0, 0], c='k', alpha=0.5)

    keeps, buys = get_optim_decisions(keep_rewards, buy_rewards)

    best_reward = get_reward(keeps, buys, keep_rewards, buy_rewards)
    print('best reward:', best_reward.item())
    worst_reward = get_reward(1 - keeps, 1 - buys, keep_rewards, buy_rewards)
    print('worst reward:', worst_reward.item())
    print('middle reward:', (best_reward.item() + worst_reward.item()) / 2)

    inp = torch.from_numpy(X_orig[:sequence_length + N - 1, :]).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)


    for e in range(n_epochs):
        optimizer.zero_grad()

        out = model(inp)
        keeps = out[:, 0]
        buys = out[:, 1]
        # print(keeps.mean().item(), keeps.std().item())
        # print(buys.mean().item(), buys.std().item())

        reward = get_reward(keeps, buys, keep_rewards, buy_rewards)
        loss = - reward

        loss.backward()
        optimizer.step()

        if (e + 1) % print_step == 0:
            keeps = torch.round(keeps)
            buys = torch.round(buys)

            buys = get_realized_decisions(keeps, buys).detach().numpy()

            wealths = get_wealths(
                X[-buys.shape[0]:, :],
                buys,
                commissions = commissions,
                spread_bull = spread if m <= 1.0 else spread_bull,
                X_bear = X_bear[:N, :],
                spread_bear = spread_bear
            )

            n_months = buys.shape[0] * aggregate_N / (24 * 30)
            wealth = wealths[-1] ** (1 / n_months)

            print(e + 1, reward.item(), wealth, wealth ** 12)

            # TODO: set log scale on when needed
            weatlhs_line.set_ydata(wealths)
            ax.set_ylim(wealths.min(), wealths.max())
            fig.canvas.draw()
            fig.canvas.flush_events()

    X_orig = X_orig[-buys.shape[0]:, :]
    plt.ioff()
    plt.clf()

    plt.plot(X_orig[:, 0] / X_orig[0, 0], c='k', alpha=0.5)
    plt.plot(wealths, c='g')
    if np.log(wealths.max() / wealths.min()) / np.log(10) > 3:
        plt.yscale('log')
    plt.show()

# TODO: make good tests for the neural network
#   - e.g. detect when the model fails
# TODO: add noise in training
#   - randomly change c and d?
# TODO: save models frequently
#   - add conditional saving based on test performance
# TODO: allow early stopping in training
#   - with smart saving, training can be resumed at any time
# TODO: add nice, automatically updating visualizations during (and after) training
#   - e.g. histograms or fillplot (must visualize uncertainty)
# TODO: add time taken to train


if __name__ == '__main__':
    lr = 0.0001
    n_epochs = 100000
    print_step = max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    # X, _ = get_recent_data(coin, 1000, 'h', 1)
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)

    aggregate_N = 3
    m = 3
    m_bear = 3

    # X_orig, X, X_bear = get_aggregated_and_multiplied_X(X, aggregate_N, m, m_bear)
    #
    # n_runs = 150
    # kappa = 20
    # optimize_reward_function(X, n_runs, kappa, aggregate_N, X_bear, m)

    c = 10 ** 0.74
    d = 10

    # plot_c_and_d(X, X_orig, c, d, aggregate_N, X_bear, m)


    model = CNN(n_features = 6, n_hidden_features_per_layer = 1)

    train(
        X,
        aggregate_N,
        m,
        m_bear,
        model,
        lr,
        n_epochs,
        print_step,
        c,
        d,
    )
