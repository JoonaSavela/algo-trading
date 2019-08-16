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
def get_labels(files, coin, n = None, l = 35, c = 10, separate = False):
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
            end_i = min(end_i, sells_idx[i])

        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        min_i = np.argmin(nearby_prices)
        min_price = nearby_prices[min_i]
        max_price = np.max(nearby_prices)

        values = np.exp(- c * (max_price / min_price - 1) * (nearby_idx - nearby_idx[min_i]) ** 2)
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
            end_i = min(end_i, buys_idx[i + 1])

        nearby_idx = np.arange(start_i, end_i)
        nearby_prices = X[nearby_idx, 0]
        max_i = np.argmax(nearby_prices)
        max_price = nearby_prices[max_i]
        min_price = np.min(nearby_prices)

        values = np.exp(- c * (max_price / min_price - 1) * (nearby_idx - nearby_idx[max_i]) ** 2)
        sells_li[start_i:end_i] = values

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



def plot_labels(files, coin, n = None):
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

    buy, sell, do_nothing = get_labels(files, coin, n, l = 35, c = 10, separate = True)

    plt.style.use('seaborn')
    #plt.plot(buys_optim, label='buys')
    plt.plot((X[:n, 0] / X[0, 0] - 1) * 100, c='k', alpha=0.5, label='price')
    plt.plot(wealths * 100, c='b', alpha=0.5, label='wealth')

    plt.plot(idx, buy, c='g', alpha=0.5)
    plt.plot(idx, sell, c='r', alpha=0.5)

    plt.axhline(0.5, c='k', linestyle=':', alpha=0.75)

    #plt.plot(idx, buys_li - sells_li, c='m', alpha=0.5)

    plt.legend()
    plt.show()



# TODO: pretrain on MLE, then train with Q-learning?
# TODO: take turns training in MLE and in Q-learning?
def train(coin, files, inputs, params, model, n_epochs, lr, batch_size, sequence_length, print_step):
    X = load_all_data(files)

    state = init_state(inputs, batch_size = batch_size)

    obs, N = get_obs_input(X, inputs, params)

    labels = get_labels(files, coin)
    labels = torch.from_numpy(labels[-N:, :])

    # discard some of the last values; their labels are bad
    N_discard = 10
    obs = obs[:-N_discard, :]
    labels = labels[:-N_discard, :]
    N -= N_discard

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(n_epochs):
        #i = torch.randint(N - sequence_length, (batch_size,))
        i = torch.zeros(batch_size).long()
        for j in range(sequence_length):
            optimizer.zero_grad()

            inp = torch.cat([state, obs[i + j, :]], dim = -1)
            print(inp.shape)
            break



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
        'capital_usd': 1,
        'capital_coin': 1,
        'timedelta': 1,
        'buy_price': 1,
        # obs
        'price': 1,
        'mus': 3,
        'std': 2,
        'ma': 2,
        'ha': 4,
    }

    params = {
        'alpha': 0.8,
        'std_window_min_max': [60, 600],
        'ma_window_min_max': [60, 1200],
    }

    sequence_length = 120

    lr = 0.001
    batch_size = 1

    # NN model definition
    model = FFN(inputs)

    n_epochs = 20
    print_step = 1#max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    #plot_labels(files, coin)

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
    )
