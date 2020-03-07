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

# TODO: try candlestick patterns with support and resistance lines?

# TODO: try getting as much (aggregated) hourly data as possible from cryptocompare
# TODO: train a NN on the (aggregated) data

def main():

    plot_performance([('sma', 5  * 60, 9, 1.4, 0.00875),
                      ('sma', 5  * 60, 9, 1.0, 0.0),
                      ('sma', 11 * 60, 4, 2.4, 0.00475)],
                      N_repeat=1)




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
