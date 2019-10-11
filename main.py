import matplotlib.pyplot as plt
import matplotlib as mpl
from data import get_and_save_all, load_data, load_all_data
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy
from utils import *
from model import *
from optimize import get_wealths
from peaks import get_peaks

# TODO:
#   - test using multiple moving averages for buy/sell signals
#       - the 1st ma is for timing
#       - the last ma is for trend
#       - the other mas are used depending on the trend

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)
    # print('Number of months:', round_to_n(len(test_files) * 2000 / (60 * 24 * 30), 3))

    # idx = np.array([0, 1, 2, 3])
    # idx = np.arange(2)
    test_files = np.array(test_files)[:10]
    X = load_all_data(test_files)

    alpha0 = 0.85
    alpha1 = 0.9975
    mus = []
    n = 10

    logit0 = p_to_logit(alpha0)
    logit1 = p_to_logit(alpha1)

    if n > 1:
        alphas = logit_to_p(np.linspace(logit0, logit1, n))
    else:
        alphas = [alpha0]
    # print(alphas)

    for i in range(n):
        mus.append(smoothed_returns(X, alphas[i]))

    rquantile = rolling_quantile(mus[-1], 1)
    # print(rquantile)
    # plt.plot(rquantile)
    # plt.show()
    # plt.hist(mus[-1], bins=50)
    # plt.show()

    N = rquantile.shape[0]

    X = X[-N:, :]
    for i in range(n):
        mus[i] = mus[i][-N:]

    li = mus[-1] > 0

    buys = np.zeros((N,)).astype(bool)
    sells = np.zeros((N,)).astype(bool)
    # sells[li] = False

    i0 = 0
    i1 = min(1, n - 1)
    # i1 = max(n - 3, 0)
    i2 = max(n - 2, 0)
    i3 = n - 1

    qth1 = 0.8

    # buys[li] = (mus[i0][li] > 0) & (mus[i2][li] > 0) & (rquantile[li] > qth1)
    # buys[~li] = (mus[i0][~li] > 0) & (mus[i1][~li] > 0) & (rquantile[~li] > qth1)

    # buys = (rquantile > qth1) & (mus[-2] > 0)

    risk_args = {
        'base_stop_loss': 0.025,
        'profit_fraction': 1.,
        'increment': 0.0025,
        'trailing_alpha': 0.85,
        'steam_th': -2.5,
        'strength': 5,
    }

    limits = {
        'base_stop_loss': [0.005, 0.05],
        'profit_fraction': [0.75, 4.],
        'increment': [0.001, 0.01], # not needed?
        'trailing_alpha': [0., 1.], # not needed?
        'steam_th': [-4., -0.],
    }


    weights = 1 / (1 - alphas)
    # print(weights)
    mus = [mus[0], np.sum(np.array(mus) * weights.reshape(-1,1), axis=0) / np.sum(weights), mus[-1]]


    buys, sells, trades = risk_management(X, mus[1], risk_args, limits, commissions = 0.0005)

    print(trades.groupby('winning')['profit'].describe())
    print(trades['profit'].sum())
    print(trades.groupby('cause')['profit'].describe())

    # print(trades.describe())
    # print(len(trades))

    # plt.hist(trades['profit'])
    # plt.show()

    # for index, row in trades.iterrows():
    #     if row['profit'] > 0:
    #         plt.plot(X[row['buy_time']:(row['sell_time'] + 1), 0] / X[row['buy_time'], 0], c='g', alpha = 0.25)
    #
    # plt.show()
    #
    # for index, row in trades.iterrows():
    #     if row['profit'] <= 0:
    #         plt.plot(X[row['buy_time']:(row['sell_time'] + 1), 0] / X[row['buy_time'], 0], c='r', alpha = 0.25)
    #
    # plt.show()

    buys = mus[0] > 0
    sells = mus[0] <= 0

    # plt.hist(mus[1], bins=500)
    # plt.show()

    # print((mus[0][li] <= 0).sum())
    # print(buys.sum(), sells.sum())

    # idx = np.arange(rquantile.shape[0])
    #
    # plt.plot(idx[buys[-rquantile.shape[0]:]], rquantile[buys[-rquantile.shape[0]:]])
    # plt.show()

    buys = buys.astype(float)
    sells = sells.astype(float)

    wealths, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0.00025
    )
    wealths += 1

    wealths1, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0
    )
    wealths1 += 1

    print(wealths[-1], wealths1[-1])
    plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    for i in range(len(mus)):
        # print(mus[i].shape)
        plt.plot(np.exp(np.cumsum(mus[i])), c='b', alpha=0.65)
    plt.plot(wealths, c='g')
    plt.plot(wealths1, c='g', alpha = 0.5)
    if np.log(wealths1[-1]) / np.log(10) > 2:
        plt.yscale('log')
    plt.show()





if __name__ == "__main__":
    main()
