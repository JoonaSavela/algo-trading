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

# TODO: try getting as much (aggregated) hourly data as possible from cryptocompare
# TODO: train a NN on the (aggregated) data

# TODO: try trailing stop loss

def main():


    plot_performance([
                        ('sma', 1 * 60, 46, 1.6, 0.0065),
                        ('sma', 1 * 60, 46, 2.6, 0.0045),
                        ('sma', 1 * 60, 46, 1.3, 0.0095),
                      ],
                      N_repeat=500)


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
    # aggregate_N = 60 * 11
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
