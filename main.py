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

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X_orig = load_all_data(test_files, 0)

    best = 0
    best_w = -1
    best_aggregate_N = -1
    best_type = ''

    for type in ['sma', 'ema']:
        for aggregate_N in range(60, 60*25, 60):
            X_all = aggregate(X_orig[np.random.randint(aggregate_N):, :], aggregate_N)
            for w in range(1, 101):
                if type == 'sma':
                    ma = np.diff(sma(X_all[:, 0] / X_all[0, 0], w))
                else:
                    alpha = 1 - 1 / w
                    ma = np.diff(ema(X_all[:, 0] / X_all[0, 0], alpha, 1.0))
                N = ma.shape[0]

                X = X_all[-N:, :]

                buys = ma > 0
                sells = ~buys

                buys = buys.astype(float)
                sells = sells.astype(float)

                wealths, _, _, _, _ = get_wealths(
                    X, buys, sells, commissions = 0.00075
                )
                wealths += 1

                n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

                wealth = wealths[-1] ** (1 / n_months)

                if wealth > best:
                    best = wealth
                    best_w = w
                    best_aggregate_N = aggregate_N
                    best_type = type

                print(wealth)

    print()
    print(best_type, best_aggregate_N // 60, best_w)
    print()

    w = best_w
    aggregate_N = best_aggregate_N
    type = best_type

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    average_wealth = 1.0
    total_months = 0

    for X in Xs:
        X_all = aggregate(X, aggregate_N)

        if type == 'sma':
            ma = np.diff(sma(X_all[:, 0] / X_all[0, 0], w))
        else:
            alpha = 1 - 1 / w
            ma = np.diff(ema(X_all[:, 0] / X_all[0, 0], alpha, 1.0))
        N = ma.shape[0]

        X = X_all[-N:, :]

        buys = ma > 0
        sells = ~buys

        buys = buys.astype(float)
        sells = sells.astype(float)

        wealths, _, _, _, _ = get_wealths(
            X, buys, sells, commissions = 0.00075
        )
        wealths += 1

        n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

        wealth = wealths[-1] ** (1 / n_months)

        print(wealth, wealth ** 12)

        plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
        plt.plot(np.cumsum(ma)+1, c='b', alpha=0.65)
        plt.plot(wealths, c='g')
        plt.show()

        average_wealth *= wealths[-1]
        total_months += n_months

    average_wealth = average_wealth ** (1 / total_months)
    print()
    print(average_wealth, average_wealth ** 12)



    # alpha0 = 0.95
    # alpha1 = 0.9975
    # print(1 / (1 - alpha0), 1 / (1 - alpha1))
    # n = 2
    #
    # logit0 = p_to_logit(alpha0)
    # logit1 = p_to_logit(alpha1)
    #
    # if n > 1:
    #     alphas = logit_to_p(np.linspace(logit0, logit1, n))
    # else:
    #     alphas = [alpha0]
    # # print(alphas)
    #
    # mus = []
    # for i in range(n):
    #     mus.append(smoothed_returns(X, alphas[i]))
    #
    # buys = np.diff(np.exp(np.cumsum(mus[0])) - np.exp(np.cumsum(mus[-1]))) > 0.0
    # # buys = buys | (mus[0] > 0)
    # sells = ~buys
    #
    # # buys = mus[0] > 0
    # # sells = mus[0] <= 0
    #
    # X = X[-buys.shape[0]:, :]
    #
    # buys = buys.astype(float)
    # sells = sells.astype(float)
    #
    # # p = 0.01
    # # buys *= p
    # # sells *= p
    #
    # wealths, _, _, _, _ = get_wealths(
    #     X, buys, sells, commissions = 0.00075
    # )
    # wealths += 1
    #
    # wealths1, _, _, _, _ = get_wealths(
    #     X, buys, sells, commissions = 0
    # )
    # wealths1 += 1
    #
    # n_months = buys.shape[0] / (60 * 24 * 30)
    #
    # wealth = wealths[-1] ** (1 / n_months)
    # wealth1 = wealths1[-1] ** (1 / n_months)
    #
    # print(wealths[-1], wealths1[-1])
    # print(wealth, wealth1)
    # plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    # for i in range(len(mus)):
    #     plt.plot(np.exp(np.cumsum(mus[i])), c='b', alpha=0.65)
    # plt.plot(wealths, c='g')
    # plt.plot(wealths1, c='g', alpha = 0.5)
    # if np.log(wealths1[-1]) / np.log(10) > 2:
    #     plt.yscale('log')
    # plt.show()
    #
    # plt.plot(np.diff(np.exp(np.cumsum(mus[0])) - np.exp(np.cumsum(mus[-1]))))
    # plt.show()





    # w = 100
    # rmax = rolling_max(X, w)
    # rmin = rolling_min(X, w)
    #
    # alpha = 0.985
    # mu = smoothed_returns(X, alpha)
    #
    # X = X[-rmax.shape[0]+1:, :]
    # rmax /= X[0, 0]
    # rmin /= X[0, 0]
    # rmax = rmax[:-1]
    # rmin = rmin[:-1]
    #
    # alpha = 0.0
    # rmax = ema(rmax, alpha, 1.0)
    # rmin = ema(rmin, alpha, 1.0)
    # X = X[-rmax.shape[0]:, :]
    # mu = mu[-X.shape[0]:]
    #
    # buys = X[:, 0] > rmax
    # sells = mu < 0
    #
    # buys = buys.astype(float)
    # sells = sells.astype(float)
    #
    # wealths, _, _, _, _ = get_wealths(
    #     X, buys, sells, commissions = 0.00015
    # )
    # wealths += 1
    #
    # wealths1, _, _, _, _ = get_wealths(
    #     X, buys, sells, commissions = 0
    # )
    # wealths1 += 1
    #
    # n_months = buys.shape[0] / (60 * 24 * 30)
    #
    # wealth = wealths[-1] ** (1 / n_months)
    # wealth1 = wealths1[-1] ** (1 / n_months)
    #
    # print(wealths[-1], wealths1[-1])
    # print(wealth, wealth1)
    #
    # plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    # plt.plot(rmax, c='g', alpha=0.65)
    # plt.plot(rmin, c='r', alpha=0.65)
    # plt.plot(np.exp(np.cumsum(mu)), c='b', alpha=0.65)
    # plt.plot(wealths, c='g')
    # plt.plot(wealths1, c='g', alpha = 0.5)
    # plt.show()





if __name__ == "__main__":
    main()
