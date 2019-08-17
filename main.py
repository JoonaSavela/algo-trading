import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import stochastic_oscillator, heikin_ashi, sma, std
from data import get_and_save_all, load_data, load_all_data
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy
from utils import get_time, round_to_n, smoothed_returns

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([2, 4, 5, -1])
    # idx = np.array([0, 1, 2, 3])
    idx = np.arange(1)
    test_files = np.array(test_files)[idx]
    X = load_all_data(test_files)
    X = X[:1200, :]

    # tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
    # c = 70
    # lips = sma(tp, 5 * c)
    # teeth = sma(tp, 8 * c)
    # jaws = sma(tp, 13 * c)
    #
    # sequence_length = jaws.shape[0]
    #
    # X = X[-sequence_length:, :]
    # lips = lips[-sequence_length:]
    # teeth = teeth[-sequence_length:]
    # jaws = jaws[-sequence_length:]
    #
    # up = (lips > teeth) & (teeth > jaws)
    # down = (lips < teeth) & (teeth < jaws)
    # neutral = ~(up | down)
    #
    # t = np.arange(sequence_length)
    #
    # plt.plot(t[up], X[up, 0], 'g.')
    # plt.plot(t[down], X[down, 0], 'r.')
    # plt.plot(t[neutral], X[neutral, 0], 'b.')
    # plt.plot(np.arange(sequence_length), lips[-sequence_length:], 'g')
    # plt.plot(np.arange(sequence_length) + 0 * c, teeth[-sequence_length:], 'r')
    # plt.plot(np.arange(sequence_length) + 0 * c, jaws[-sequence_length:], 'b')
    # plt.show()

    returns = X[1:, 0] / X[:-1, 0] - 1

    alpha = 0.8
    mus = []

    r = returns

    N = 3
    for i in range(N):
        r = smoothed_returns(np.cumprod(r + 1).reshape(-1, 1), alpha = alpha)
        mus.append(r)

    n = mus[-1].shape[0]
    buys = []
    sells = []

    for i in range(N - 1):
        buys.append(np.cumprod(mus[i][-n:] + 1) > np.cumprod(mus[i + 1][-n:] + 1))
        sells.append(np.cumprod(mus[i][-n:] + 1) < np.cumprod(mus[i + 1][-n:] + 1))

    buys = np.stack(buys)
    sells = np.stack(sells)

    buys = np.all(buys, axis=0)
    sells = np.all(sells, axis=0)

    buys = np.diff(buys)
    sells = np.diff(sells)
    idx = np.arange(1, n)
    buys = idx[buys]
    sells = idx[sells]

    i = 0
    j = 0

    new_buys = []
    new_sells = []

    while i < buys.shape[0] and j < sells.shape[0]:
        new_buys.append(buys[i])
        i += 1


    #mus1 = smoothed_returns(X, alpha = alpha * 2 - alpha ** 2)


    #plt.plot(returns, label='returns')
    #plt.plot(mus, label='mean')
    #plt.axhline(c='k', alpha=0.5)
    plt.plot(X[-n:, 0] / X[0, 0] - 1, c='k', alpha = 0.3, label='price')

    for i in range(N):
        approx_price = np.cumprod(mus[i][-n:] + 1)
        plt.plot(approx_price / approx_price[0] - 1, label=str(i))

    for x in buys:
        plt.axvline(x, c='g', alpha=0.5)

    for x in sells:
        plt.axvline(x, c='r', alpha=0.5)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
