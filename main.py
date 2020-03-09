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
from optimize import *
from peaks import get_peaks
from tqdm import tqdm
import binance
from binance.client import Client
from binance.enums import *
from keys import binance_api_key, binance_secret_key
from parameter_search import *
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# TODO: train a NN on the (aggregated) data

# TODO: try trailing stop loss

# TODO: try shorting

def plot_performance_short(params_list, N_repeat = 1):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    c_list = ['g', 'c', 'm', 'r']

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for i, params in enumerate(params_list):
        aggregate_N, w, m, p = params
        print(aggregate_N // 60, w, m, p)

        total_log_wealths = []
        total_months = []

        for n in tqdm(range(N_repeat), disable = N_repeat == 1):
            prev_price = 1.0
            count = 0
            total_wealth1 = 1.0
            total_months1 = 0

            for X in Xs:
                rand_N = np.random.randint(aggregate_N)
                if rand_N > 0:
                    X = X[:-rand_N, :]
                X_agg = aggregate(X, aggregate_N)

                buys, sells, N = get_buys_and_sells(X_agg, w)

                X_agg = X_agg[-N:, :]
                X = X[-aggregate_N * N:, :]

                wealths, _, _, _, _ = get_wealths_short(
                    X_agg, buys, sells, commissions = 0.00075
                )

                n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

                wealth = wealths[-1] ** (1 / n_months)

                if N_repeat == 1:
                    print(wealth, wealth ** 12)

                t = np.arange(N) + count
                t *= aggregate_N
                count += N

                if N_repeat == 1:
                    if i == 0:
                        plt.plot(t, X_agg[:, 0] / X_agg[0, 0] * prev_price, c='k', alpha=0.5)
                    # plt.plot(t, (np.cumsum(ma) + 1) * prev_price, c='b', alpha=0.65 ** i)
                    plt.plot(t, wealths * total_wealth1, c=c_list[i % len(c_list)], alpha=0.9 / np.sqrt(N_repeat))

                total_wealth1 *= wealths[-1]
                prev_price *= X[-1, 0] / X[0, 0]
                total_months1 += n_months

            total_log_wealths.append(np.log(total_wealth1))
            total_months.append(total_months1)

        total_wealth = np.exp(np.sum(total_log_wealths) / np.sum(total_months))
        print()
        print(total_wealth, total_wealth ** 12)
        print()

        if N_repeat > 1:
            plt.hist(
                np.array(total_log_wealths) / np.array(total_months),
                np.sqrt(N_repeat).astype(int),
                color=c_list[i % len(c_list)],
                alpha=0.9 / np.sqrt(len(params_list))
            )

    plt.show()

def main():

    plot_performance_short([
                        (1 * 60, 46, 1.6, 0.0065),
                        (5 * 60, 9, 1.3, 0.00875),
                      ],
                      N_repeat=1000)

    # plot_performance([
    #                     (1 * 60, 46, 1.6, 0.0065),
    #                     (5 * 60, 9, 1.3, 0.00875),
    #                   ],
    #                   N_repeat=1)


    # plt.style.use('seaborn')
    #
    # test_files = glob.glob('data/ETH/*.json')
    # test_files.sort(key = get_time)
    #
    # X = load_all_data(test_files, 0)
    #
    # aggregate_N = 60 * 1
    #
    # # rand_N = np.random.randint(aggregate_N)
    # # if rand_N > 0:
    # #     X = X[:-rand_N, :]
    # X_agg = aggregate(X, aggregate_N)
    #
    # ma = np.diff(sma(X_agg[:, 0] / X_agg[0, 0], 2))
    # stoch = stochastic_oscillator(X_agg, 14)
    # N = min(stoch.shape[0], ma.shape[0])
    # stoch = stoch[-N:]
    # ma = ma[-N:]
    #
    # th = 0.1
    # buys = (stoch < th) #& (ma > 0)
    # sells = (stoch > 1 - th) #& (ma < 0)
    #
    # buys = buys.astype(float)
    # sells = sells.astype(float)
    #
    # X_agg = X_agg[-N:, :]
    #
    # wealths, _, _, _, _ = get_wealths(
    #     X_agg, buys, sells, commissions = 0.00075
    # )
    #
    # n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)
    #
    # wealth = wealths[-1] ** (1 / n_months)
    #
    # print(wealth, wealth ** 12)
    #
    # plt.plot(X_agg[:, 0] / X_agg[0, 0], c='k', alpha=0.5)
    # plt.plot(wealths, c='g')
    # plt.show()


    # plt.style.use('seaborn')
    #
    # test_files = glob.glob('data/ETH/*.json')
    # test_files.sort(key = get_time)
    #
    # X = load_all_data(test_files, 0)
    #
    # aggregate_N = 60 * 1
    #
    # X_agg = aggregate(X, aggregate_N)
    #
    # X_agg_flat = X_agg[:, :4].reshape(-1)
    #
    # weights = X_agg[:, 4] / np.sum(X_agg[:, 4]) / 4
    # weights = np.tile(weights, 4)
    #
    # x_grid = np.linspace(np.min(X_agg_flat), np.max(X_agg_flat), 10000)
    #
    # # kde = gaussian_kde(X_agg_flat, bw_method=0.2)
    # # old_kde = kde.evaluate(x_grid)
    # #
    # # new_kde = np.zeros(10000)
    # #
    # # for i, x in enumerate(tqdm(X_agg_flat, disable = False)):
    # #     min_i = np.argmin(np.abs(x_grid - x))
    # #     new_kde[min_i] += old_kde[min_i] * weights[i // 4]
    #
    # new_X = np.random.choice(X_agg_flat, X_agg_flat.shape[0], p=weights)
    #
    # kde = gaussian_kde(new_X, bw_method=0.1)
    #
    # peaks, _ = find_peaks(kde.evaluate(x_grid))
    # print(x_grid[peaks])
    #
    # plt.plot(x_grid, kde.evaluate(x_grid))
    # plt.show()
    #
    # plt.plot(X_agg[:, 0])
    # plt.hlines(x_grid[peaks], 0, X_agg.shape[0])
    # plt.show()




if __name__ == "__main__":
    main()
