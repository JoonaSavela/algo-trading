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

    # idx = np.arange(1, len(test_files))
    # li = np.diff(np.array(list(map(lambda x: get_time(x), test_files)))) != 120000
    #
    # points = [0]
    # for p in idx[li]:
    #     points.append(p)
    # points.append(len(test_files))
    #
    # print(points)
    # count = 0
    #
    # for start, end in zip(points[:-1], points[1:]):
    #     print(start, end)
    #     X = load_all_data(test_files[start:end])
    #     N = X.shape[0]
    #
    #     plt.plot(np.arange(N) + count, X[:, 0], c='k', alpha=0.5)
    #     plt.plot([count], X[0, 0], 'g.', markersize=12)
    #     plt.plot([count + N - 1], X[-1, 0], 'r.', markersize=12)
    #     count += N
    # plt.show()

    X = load_all_data(test_files, 0)
    if isinstance(X, list):
        count = 0
        for x in X:
            N = x.shape[0]
            plt.plot(np.arange(N) + count, x[:, 0], c='k', alpha=0.5)
            count += N
    else:
        plt.plot(X[:, 0], c='k', alpha=0.5)
    plt.show()

    return

    # idx = np.array([0, 1, 2, 3])
    # idx = np.arange(2)
    test_files = np.array(test_files)[:2]
    X = load_all_data(test_files)

    w0 = 5
    w1 = 500
    n = 1

    if n > 1:
        ws = np.round(np.linspace(w0, w1, n)).astype(int)
    else:
        ws = [w0]

    print(ws)
    N = X.shape[0] - max(ws) + 1 - 1

    mas = []
    for w in ws:
        mas.append(np.diff(sma(X[:, 0] / X[0, 0], w))[-N:])

    X = X[-N:, :]

    buys = mas[0] > 0
    sells = ~buys

    buys = buys.astype(float)
    sells = sells.astype(float)

    wealths, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0.00075
    )
    wealths += 1

    wealths1, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0
    )
    wealths1 += 1

    n_months = buys.shape[0] / (60 * 24 * 30)

    wealth = wealths[-1] ** (1 / n_months)
    wealth1 = wealths1[-1] ** (1 / n_months)

    print(wealths[-1], wealths1[-1])
    print(wealth, wealth1)


    plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    for i in range(len(mas)):
        plt.plot(np.cumsum(mas[i])+1, c='b', alpha=0.65)
    plt.plot(wealths, c='g')
    plt.plot(wealths1, c='g', alpha = 0.5)
    if np.log(wealths1[-1]) / np.log(10) > 2:
        plt.yscale('log')
    plt.show()



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
