# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from model import *
from data import load_all_data
from utils import *
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from optimize import get_wealths

# TODO: make labels skewed to the right
# TODO: reduce repetition
# TODO: get the capitals as well, feed them to the initial states
#       - get also buy prices and timedeltas?
def get_labels(files, coin, n = None, l = 75, c = 2, skew = 0.8, separate = False):
    X = load_all_data(files)

    if n is None:
        n = X.shape[0]

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    diffs = np.diff(np.concatenate([np.array([0]), buys_optim, np.array([0])]))
    idx = np.arange(n + 1)

    buys_li = diffs == 1
    sells_li = diffs == -1

    buys_idx = idx[buys_li]
    sells_idx = idx[sells_li]

    buys_li = buys_li.astype(float)
    sells_li = sells_li.astype(float)

    for i in range(buys_idx.shape[0]):
        buy_i = buys_idx[i]

        start_i = max(0, buy_i - l)
        end_i = min(n - 1, buy_i + l)

        if i > 0:
            start_i = max(start_i, (buy_i + buys_idx[i - 1]) // 2)
            start_i = max(start_i, sells_idx[i - 1])

        if i < buys_idx.shape[0] - 1:
            end_i = min(end_i, (buy_i + buys_idx[i + 1]) // 2)
            end_i = min(end_i, (buy_i + sells_idx[i]) // 2)

        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        min_i = np.argmin(nearby_prices)
        min_price = nearby_prices[min_i]
        max_price = np.max(nearby_prices)

        values = (nearby_idx - nearby_idx[min_i]) * (1 - skew * (nearby_idx > nearby_idx[min_i]).astype(float))
        values = np.exp(- c * (max_price / min_price - 1) * values ** 2)
        buys_li[start_i:end_i] = values

    for i in range(sells_idx.shape[0]):
        sell_i = sells_idx[i]

        start_i = max(0, sell_i - l)
        end_i = min(n - 1, sell_i + l)

        if i > 0:
            start_i = max(start_i, (sell_i + sells_idx[i - 1]) // 2)
            start_i = max(start_i, buys_idx[i])

        if i < sells_idx.shape[0] - 1:
            end_i = min(end_i, (sell_i + sells_idx[i + 1]) // 2)
            end_i = min(end_i, (sell_i + buys_idx[i + 1]) // 2)

        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        max_i = np.argmax(nearby_prices)
        max_price = nearby_prices[max_i]
        min_price = np.min(nearby_prices)

        values = (nearby_idx - nearby_idx[max_i]) * (1 - skew * (nearby_idx > nearby_idx[max_i]).astype(float))
        values = np.exp(- c * (max_price / min_price - 1) * values ** 2)
        sells_li[start_i:end_i] = values

    # buys_optim = (buys_li == 1)

    diffs_li = buys_li - sells_li

    buy = np.zeros(diffs_li.shape[0])
    sell = np.zeros(diffs_li.shape[0])

    li = diffs_li > 0
    buy[li] = diffs_li[li]

    li = diffs_li < 0
    sell[li] = -diffs_li[li]

    buy = buy[:-1]
    sell = sell[:-1]

    do_nothing = 1 - buy - sell

    if separate:
        return buy, sell, do_nothing

    labels = np.stack([buy, sell, do_nothing], axis = 1)

    return labels



def plot_labels(files, coin, n = 600):
    X = load_all_data(files)

    if n is None:
        n = X.shape[0]

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    wealths, _, _, _, _ = get_wealths(
        X[:n, :], buys_optim
    )

    idx = np.arange(n)

    buy, sell, do_nothing = get_labels(files, coin, n, separate = True)

    plt.style.use('seaborn')
    #plt.plot(buys_optim, label='buys')
    # plt.plot(X[:n, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    # plt.plot(wealths + 1, c='b', alpha=0.5, label='wealth')
    plt.plot((X[:n, 0] / X[0, 0] - 1) * 100, c='k', alpha=0.5, label='price')
    plt.plot(wealths * 100, c='b', alpha=0.5, label='wealth')

    plt.plot(idx, buy, c='g', alpha=0.5)
    plt.plot(idx, sell, c='r', alpha=0.5)

    plt.axhline(0.5, c='k', linestyle=':', alpha=0.75)

    #plt.plot(idx, buys_li - sells_li, c='m', alpha=0.5)

    # plt.yscale('log')
    plt.legend()
    plt.show()


def update_state(out, state, price, ma_ref, commissions, eps = 0.1):
    capital_usd = state[:, 0]
    capital_coin = state[:, 1]
    timedelta = state[:, 2]
    buy_price = state[:, 3]

    BUY = out[:, 0]
    SELL = out[:, 1]
    DO_NOTHING = out[:, 2]

    capital_coin_in_usd = capital_coin * price
    total_wealth = capital_usd + capital_coin_in_usd

    li1 = (DO_NOTHING < 1 - eps) & (capital_usd / total_wealth > eps) & (BUY * capital_usd > SELL * capital_coin_in_usd)
    li2 = (DO_NOTHING < 1 - eps) & (capital_coin_in_usd / total_wealth > eps) & (BUY * capital_usd < SELL * capital_coin_in_usd)
    li3 = timedelta != -1

    timedelta[li3] += 1

    timedelta[li1] = 0
    buy_price[li1] = price[li1] / ma_ref[li1]

    timedelta[li2] = -1
    buy_price[li2] = -1

    amount_usd_buy = capital_usd * BUY
    amount_coin_buy = amount_usd_buy / price * (1 - commissions)

    amount_coin_sell = capital_coin * SELL
    amount_usd_sell = amount_coin_sell * price * (1 - commissions)

    capital_coin += amount_coin_buy - amount_coin_sell
    capital_usd += amount_usd_sell - amount_usd_buy

    state = torch.stack([capital_usd, capital_coin, timedelta, buy_price], dim = 1)

    return state


# TODO: pretrain on MLE, then train with Q-learning?
# TODO: take turns training in MLE and in Q-learning?
def train(coin, files, inputs, params, model, n_epochs, lr, batch_size, sequence_length, print_step, commissions):
    X = load_all_data(files)

    obs, N, ma_ref = get_obs_input(X, inputs, params)
    X = X[-N:, :]

    labels = get_labels(files, coin)

    labels = torch.from_numpy(labels[-N:, :]).type(torch.float32)

    prices = torch.from_numpy(X[:, 0]).type(torch.float32)

    # discard some of the last values; their labels are bad
    N_discard = 20
    obs = obs[:-N_discard, :]
    labels = labels[:-N_discard, :]
    X = X[:-N_discard, :]
    ma_ref = ma_ref[:-N_discard]
    prices = prices[:-N_discard]
    N -= N_discard

    N_test = sequence_length
    # print('N test: {}'.format(N_test))
    N -= N_test

    obs, obs_test = obs[:N, :], obs[N:, :]
    labels, labels_test = labels[:N, :], labels[N:, :]
    X, X_test = X[:N, :], X[N:, :]
    ma_ref, ma_ref_test = ma_ref[:N], ma_ref[N:]
    prices, prices_test = prices[:N], prices[N:]

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(n_epochs):
        i = torch.randint(N - sequence_length, (batch_size,))
        # i = torch.zeros(batch_size).long()# + 1

        # state = init_state(inputs, batch_size = batch_size)
        model.init_state()

        losses = []

        optimizer.zero_grad()

        # for j in range(sequence_length):

        inp = []
        for ii in i:
            inp.append(obs[ii:ii+sequence_length, :])
        inp = torch.stack(inp, dim=1).detach()
        # inp = torch.cat([state, obs[i + j, :]], dim = -1).unsqueeze(0).detach()

        out = model(inp)
        # target = labels[i + j, :]
        target = []
        for ii in i:
            target.append(labels[ii:ii+sequence_length, :])
        target = torch.stack(target, dim=1)

        loss = criterion(out, target)

        loss.backward()
        losses.append(loss.item())

        # state = update_state(out.detach(), state.detach(), prices[i + j], ma_ref[i + j], commissions)

        optimizer.step()

        if e % print_step == 0:
            benchmark_profits = []
            profits = []
            # optim_profits = []

            for b in range(batch_size):
                benchmark_profits.append(X[i[b] + sequence_length - 1, 0] / X[i[b], 0])

                buys = out[:, b, 0].detach().numpy()
                sells = out[:, b, 1].detach().numpy()
                wealths, _, _, _, _ = get_wealths(
                    X[i[b]:i[b]+sequence_length, :], buys, sells, commissions = commissions
                )
                profits.append(wealths[-1] + 1)

            n_round = 4

            # print('[Epoch: {}/{}] [Loss: {}] [Avg. Profit: {}] [Benchmark Profit: {}]'.format(
            #     e,
            #     n_epochs,
            #     round_to_n(torch.tensor(losses).mean().item(), n_round),
            #     round_to_n(state[:, :2].sum(dim = 1).prod().item() ** (1 / batch_size), n_round),
            #     round_to_n(torch.tensor(benchmark_profits).prod().item() ** (1 / batch_size), n_round),
            # ))
            print('[Epoch: {}/{}] [Loss: {}] [Avg. Profit: {}] [Benchmark Profit: {}]'.format(
                e,
                n_epochs,
                round_to_n(torch.tensor(losses).mean().item(), n_round),
                round_to_n(torch.tensor(profits).prod().item() ** (1 / batch_size), n_round),
                round_to_n(torch.tensor(benchmark_profits).prod().item() ** (1 / batch_size), n_round),
            ))
            # print(out[0])

    model.eval()
    model.init_state(1)
    inp = obs_test.unsqueeze(1)
    out = model(inp)

    buys = out[:, 0, 0].detach().numpy()
    sells = out[:, 0, 1].detach().numpy()

    initial_usd = 1000

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X_test, buys, sells, initial_usd = initial_usd, commissions = commissions
    )

    print(wealths[-1] + 1, capital_usd / initial_usd, capital_coin * X_test[0, 0] / initial_usd)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X_test[:, 0] / X_test[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].legend()
    plt.show()



if __name__ == '__main__':
    commissions = 0.00075

    # inputs:
    #   note: all prices (and stds) are relative to a running average price
    #   - state:
    #       - capital_usd
    #       - capital_coin (in usd)
    #       - time since bought (or -1)
    #       - price when bought (or -1)
    #       - other data at time of bought?
    #   - obs:
    #       - close, high, low, open (not all?)
    #       - running std, or bollinger band width
    #       - sma, alligator stuff?
    #       - smoothed returns
    #       - stoch
    #       - ha
    inputs = {
        # states
        # 'capital_usd': 1,
        # 'capital_coin': 1,
        # 'timedelta': 1,
        # 'buy_price': 1,
        # obs
        'price': 1,
        'mus': 3,
        'std': 3,
        'ma': 3,
        'ha': 3,
        'stoch': 3,
    }

    params = {
        'alpha': 0.8,
        'std_window_min_max': [30, 2000],
        'ma_window_min_max': [30, 2000],
        'stoch_window_min_max': [30, 2000],
    }

    sequence_length = 500

    lr = 0.0005
    batch_size = 32

    # NN model definition
    model = FFN(inputs, batch_size)

    n_epochs = 1500
    print_step = max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    # plot_labels(files, coin)

    start_time = time.time()

    train(
        coin = coin,
        files = files,
        inputs = inputs,
        params = params,
        model = model,
        n_epochs = n_epochs,
        lr = lr,
        batch_size = batch_size,
        sequence_length = sequence_length,
        print_step = print_step,
        commissions = commissions,
    )

    print('Time taken: {} seconds'.format(round_to_n(time.time() -start_time, 3)))
