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

# TODO: try candlestick patterns

def main():
    plot_performance([('sma', 5  * 60, 9, 1.4, 0.00875),
                      ('sma', 11 * 60, 4, 2.4, 0.00475)])

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
