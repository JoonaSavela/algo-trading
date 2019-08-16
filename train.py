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

def get_labels(files, coin, n = None, l = 35, c = 10):
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

    do_nothing = 1 - buy - sell

    return buy, sell, do_nothing



def plot_labels(files, coin, n = 800):
    X = load_all_data(files)

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

    idx = np.arange(n + 1)

    buy, sell, do_nothing = get_labels(files, coin, n, l = 35, c = 10)

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
def train(coin, files, model, n_epochs, lr):
    X = load_all_data(files)

    for e in range(n_epochs):
        break


if __name__ == '__main__':
    commissions = 0.00075

    # inputs:
    #   - state:
    #       -
    #   - obs:
    #       -
    inputs = {
        'close',

    }

    input_size = len(inputs)
    lr = 0.001

    # NN model definition
    model = FFN(input_size)

    n_epochs = 20
    print_step = 1#max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    train(
        coin = coin,
        files = files,
        model = model,
        n_epochs = n_epochs,
        lr = lr,
    )

    plot_labels(files, coin)
