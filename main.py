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
from optimize import get_wealths, get_wealths_limit
from peaks import get_peaks
from tqdm import tqdm
import binance
from binance.client import Client
from binance.enums import *
from keys import binance_api_key, binance_secret_key

# TODO: try candlestick patterns

def get_wealths_oco(X, X_agg, aggregate_N, p, m, buys, sells = None, initial_usd = 1000, initial_coin = 0, commissions = 0.00075, verbose = True):
    if sells is None:
        sells = 1 - buys
    capital_usd = initial_usd
    capital_coin = initial_coin

    wealths = [0] * X_agg.shape[0]
    buy_amounts = []
    sell_amounts = []

    buy_lim_count = 0
    buy_stop_count = 0
    sell_lim_count = 0
    sell_stop_count = 0

    for i in range(X_agg.shape[0] - 1):
        BUY, SELL = buys[i], sells[i]
        price = X_agg[i, 0]
        buy_price = price
        sell_price = price

        if BUY > 0.0 and capital_usd > 0.0:
            idx = np.arange((i + 1) * aggregate_N, (i + 2) * aggregate_N)
            # TODO: different profit ratio than 1:1
            li_limit = X[idx, 2] <= price * np.exp(-m*p)
            li_stop = X[idx, 1] > price * np.exp(p)

            if li_limit.any():
                first_limit_idx = idx[li_limit][0]
            else:
                first_limit_idx = np.inf

            if li_stop.any():
                first_stop_idx = idx[li_stop][0]
            else:
                first_stop_idx = np.inf

            if (first_limit_idx == first_stop_idx and first_limit_idx != np.inf) or \
                    (first_stop_idx < first_limit_idx):
                buy_price = price * np.exp(p)
                buy_stop_count += 1
            elif first_limit_idx < first_stop_idx:
                buy_price = price * np.exp(-m*p)
                buy_lim_count += 1
            else:
                BUY = 0.0

        if SELL > 0.0 and capital_coin > 0.0:
            idx = np.arange((i + 1) * aggregate_N, (i + 2) * aggregate_N)
            # TODO: different profit ratio than 1:1
            li_limit = X[idx, 1] >= price * np.exp(m*p)
            li_stop = X[idx, 2] < price * np.exp(-p)

            if li_limit.any():
                first_limit_idx = idx[li_limit][0]
            else:
                first_limit_idx = np.inf

            if li_stop.any():
                first_stop_idx = idx[li_stop][0]
            else:
                first_stop_idx = np.inf

            if (first_limit_idx == first_stop_idx and first_limit_idx != np.inf) or \
                    (first_stop_idx < first_limit_idx):
                sell_price = price * np.exp(-p)
                sell_stop_count += 1
            elif first_limit_idx < first_stop_idx:
                sell_price = price * np.exp(m*p)
                sell_lim_count += 1
            else:
                SELL = 0.0

        amount_coin_buy = BUY * capital_usd / buy_price * (1 - commissions)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * sell_price * (1 - commissions)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        buy_amounts.append(amount_usd_buy)
        sell_amounts.append(amount_usd_sell)

        wealths[i] = capital_usd + capital_coin * price

    if verbose:
        print(buy_lim_count, buy_stop_count)
        print(sell_lim_count, sell_stop_count)

    wealths[-1] = wealths[-2]

    wealths = np.array(wealths) / wealths[0]

    return wealths, capital_usd, capital_coin, buy_amounts, sell_amounts

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)

    aggregate_N = 60 * 11
    w = 4

    X_agg = aggregate(X, aggregate_N)

    ma = np.diff(sma(X_agg[:, 0] / X_agg[0, 0], w))
    N = ma.shape[0]

    X_agg = X_agg[-N:, :]
    X = X[-aggregate_N * N:, :]

    best_p = 0.0
    best_m = 0
    best_wealth = -1

    for m in np.arange(1.0, 3.0, 0.1):
        for p in np.arange(0, 0.01, 0.00025):

            buys = ma > 0
            sells = ~buys

            buys = buys.astype(float)
            sells = sells.astype(float)

            wealths, _, _, _, _ = get_wealths_oco(
                X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = False
            )

            n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

            wealth = wealths[-1] ** (1 / n_months)

            if wealth > best_wealth:
                best_wealth = wealth
                best_p = p
                best_m = m

            print(m, p, wealth, wealth ** 12)

    print()
    # 2, 0.005
    # 2.4, 0.00475
    print(best_m, best_p, best_wealth, best_wealth ** 12)
    print()

    p = best_p
    m = best_m

    # commissions = 0.0

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    total_wealth = 1.0
    total_months = 0
    count = 0
    prev_price = 1.0

    for X in Xs:
        X_agg = aggregate(X, aggregate_N)

        ma = np.diff(sma(X_agg[:, 0] / X_agg[0, 0], w))
        N = ma.shape[0]

        X_agg = X_agg[-N:, :]
        X = X[-aggregate_N * N:, :]

        buys = ma > 0
        sells = ~buys

        buys = buys.astype(float)
        sells = sells.astype(float)

        wealths, _, _, _, _ = get_wealths_oco(
            X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = False
        )

        n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

        wealth = wealths[-1] ** (1 / n_months)

        print(wealth, wealth ** 12)

        t = np.arange(N) + count
        count += N

        plt.plot(t, X_agg[:, 0] / X_agg[0, 0] * prev_price, c='k', alpha=0.5)
        plt.plot(t, (np.cumsum(ma) + 1) * prev_price, c='b', alpha=0.65)
        plt.plot(t, wealths * total_wealth, c='g')

        total_wealth *= wealths[-1]
        prev_price *= X[-1, 0] / X[0, 0]
        total_months += n_months

    total_wealth = total_wealth ** (1 / total_months)
    print()
    print(total_wealth, total_wealth ** 12)

    plt.show()

    # buys = ma > 0
    # sells = ~buys
    #
    # buys = buys.astype(float)
    # sells = sells.astype(float)
    #
    # wealths, _, _, _, _ = get_wealths_oco(
    #     X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075
    # )
    #
    # n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)
    #
    # wealth = wealths[-1] ** (1 / n_months)
    #
    # print(m, p, wealth, wealth ** 12)
    #
    # plt.plot(X_agg[:, 0] / X_agg[0, 0], c='k', alpha=0.5)
    # plt.plot(np.cumsum(ma) + 1, c='b', alpha=0.65)
    # plt.plot(wealths, c='g')
    # plt.show()

    # high_diff = X[:, 1] / X[:, 0] - 1
    # print(np.median(high_diff), np.mean(high_diff))
    # N = high_diff.shape[0]
    # N2 = np.round(1 * N).astype(int)
    #
    # high_diff = np.sort(high_diff)[:N2]
    #
    # low_diff = X[:, 2] / X[:, 0] - 1
    # print(np.median(low_diff), np.mean(low_diff))
    #
    # low_diff = np.sort(low_diff)[:N2]
    #
    # print(np.mean(np.min(high_diff) == high_diff))
    # print(np.mean(np.max(low_diff) == low_diff))
    #
    # plt.hist(high_diff, 50)
    # plt.show()
    #
    # plt.hist(low_diff, 50)
    # plt.show()




if __name__ == "__main__":
    main()
