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

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([0, 1, 2, 3])
    idx = np.arange(2)
    test_files = np.array(test_files)[idx]
    X = load_all_data(test_files)
    # X = X[:1200, :]

    inputs = {
        # states
        # 'capital_usd': 1,
        # 'capital_coin': 1,
        # 'timedelta': 1,
        # 'buy_price': 1,
        # obs
        'price': 4,
        'mus': 3,
        'std': 3,
        'ma': 3,
        'ha': 4,
        'stoch': 3,
    }

    params = {
        'alpha': 0.8,
        'std_window_min_max': [30, 2000],
        'ma_window_min_max': [30, 2000],
        'stoch_window_min_max': [30, 2000],
    }

    sequence_length = 60

    signal_model = FFN(inputs, 1, use_lstm = True, Qlearn = False)
    signal_model.load_state_dict(torch.load('models/' + signal_model.name + '.pt'))
    signal_model.eval()

    returns = X[1:, 0] / X[:-1, 0] - 1

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]
    returns = returns[-N:]

    signal_model.init_state()

    inp = obs.unsqueeze(1)
    out = signal_model(inp).squeeze(1)

    buys = out[:, 0].detach().numpy()
    sells = out[:, 1].detach().numpy()

    sequence_length = min(sequence_length, N)

    X = X[:sequence_length, :]
    returns = returns[:sequence_length]
    sells = sells[:sequence_length]
    buys = buys[:sequence_length]

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    n = 10
    # start = 5
    for start in range(15, 25):
        x = np.arange(start, start + n)
        y = sells[start:start+n]

        z = np.polyfit(x, y, 2)
        a, b, c = tuple(z)
        print(start, round(- b / (2 * a)))

        fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

        ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
        ax[0].legend()

        #ax[1].plot(buys, c='g', alpha=0.5, label='buy')
        ax[1].plot(sells, c='r', alpha=0.5, label='sell')
        ax[1].plot(x, y, 'ko', alpha=0.5, label='y')
        x = np.arange(start, start + n * 3 // 2)
        ax[1].plot(x, np.polyval(z, x), c='b', alpha=0.5, label='poly')
        ax[1].legend()
        plt.show()


if __name__ == "__main__":
    main()
