import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
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
from tqdm import tqdm
import binance
from binance.client import Client
from binance.enums import *
from keys import binance_api_key, binance_secret_key

# TODO: check whether limit orders would work
# TODO: try out smoothed returns
# TODO: move this to a different file
# TODO: make a separate function for calculating buys and sells

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)


    # w = 4
    # aggregate_N = 11
    #
    # X, _ = get_recent_data('ETH', 200, 'h', aggregate_N)
    #
    # ma = np.diff(sma(X[:, 0] / X[0, 0], w))
    # N = ma.shape[0]
    #
    # X = X[-N:, :]
    #
    # buys = ma > 0
    # sells = ~buys
    #
    # buys = buys.astype(float)
    # sells = sells.astype(float)
    #
    # wealths, _, _, _, _ = get_wealths(
    #     X, buys, sells, commissions = 0.00075
    # )
    # wealths += 1
    #
    # n_months = buys.shape[0] * aggregate_N * 60 / (60 * 24 * 30)
    #
    # wealth = wealths[-1] ** (1 / n_months)
    #
    # print(wealth, wealth ** 12)
    #
    # plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    # plt.plot(np.cumsum(ma) + 1, c='b', alpha=0.65)
    # plt.plot(wealths, c='g')
    # plt.show()

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



if __name__ == "__main__":
    main()
