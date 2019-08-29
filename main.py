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

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([0, 1, 2, 3])
    # idx = np.arange(2)
    test_files = np.array(test_files)[:10]
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

    sequence_length = 750

    signal_model = FFN(inputs, 1, use_lstm = True, Qlearn = False, use_tanh = False)
    signal_model.load_state_dict(torch.load('models/' + signal_model.name + '.pt'))
    signal_model.eval()

    model = FFN(dict(n_inputs=3), 1, use_tanh = True, hidden_size = 16)
    model.load_state_dict(torch.load('models/' + model.name + '.pt'))
    model.eval()

    alpha = 0.#75
    mus = smoothed_returns(X, alpha=alpha)
    mus = smoothed_returns(np.cumprod(mus + 1).reshape(-1, 1), alpha=alpha)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]
    mus = mus[-N:]

    sequence_length = min(sequence_length, N)

    X = X[:sequence_length, :]
    obs = obs[:sequence_length, :]
    mus = mus[:sequence_length]

    signal_model.init_state()
    model.init_state()

    inp = obs.unsqueeze(1)
    signal_out = signal_model(inp)

    out = model(signal_out).detach().numpy()

    buys = np.zeros(sequence_length)
    sells = np.zeros(sequence_length)

    li = out[:, 0, 0] > 0
    buys[li] = out[li, 0, 0]

    li = out[:, 0, 0] < 0
    sells[li] = -out[li, 0, 0]

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    # buys = buys ** 2
    # sells = sells ** 2

    idx = np.arange(1, sequence_length)

    th = 0.5
    buys_li = (buys[1:] > th) & (np.diff(buys) < 0)
    buy_peaks = idx[buys_li]

    sells_li = (sells[1:] > th) & (np.diff(sells) < 0)
    sell_peaks = idx[sells_li]

    d = 0
    buy_peaks += d
    sell_peaks += d
    buy_peaks = buy_peaks[buy_peaks < sequence_length]
    sell_peaks = sell_peaks[sell_peaks < sequence_length]

    owns1 = np.zeros((sequence_length,))

    for peak in buy_peaks:
        sell_peak = sell_peaks[sell_peaks > peak]
        sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
        owns1[peak:sell_peak] = 1

    buys1 = owns1 == 1
    sells1 = owns1 == 0

    buys2 = mus > 0
    sells2 = mus < 0

    buys1 = buys1 & buys2
    sells1 = sells1 & sells2

    buys1 = buys1.astype(float)
    sells1 = sells1.astype(float)

    initial_usd = 1000

    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X, buys1, sells1, initial_usd = initial_usd,
    )

    print(wealths[-1] + 1)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks')
    ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks')
    ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    ax[0].legend()

    ax[1].plot(signal_out[:, 0, 1].detach().numpy(), c='k', alpha=0.5, label='signal')
    ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].plot(sell_peaks, sells[sell_peaks], 'ro', alpha=0.7, label='sell peaks')
    ax[1].plot(buy_peaks, buys[buy_peaks], 'go', alpha=0.7, label='buy peaks')
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
