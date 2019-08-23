import torch
from scipy.signal import find_peaks
import glob
from utils import *
from data import *
from model import *
import numpy as np
from optimize import get_wealths
import matplotlib.pyplot as plt

def plot_peaks(coin, files, inputs, params, model, sequence_length):
    X = load_all_data(files)

    alpha = 0.#35
    mus = smoothed_returns(X, alpha=alpha)
    mus = smoothed_returns(np.cumprod(mus + 1).reshape(-1, 1), alpha=alpha)

    ha = heikin_ashi(X)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]
    ha = ha[-N:, :]
    mus = mus[-N:]

    X = X[:sequence_length, :]
    ha = ha[:sequence_length, :]
    obs = obs[:sequence_length, :]
    mus = mus[:sequence_length]

    model.init_state()

    inp = obs.unsqueeze(1)
    out = model(inp).squeeze(1)

    buys = out[:, 0].detach().numpy()
    sells = out[:, 1].detach().numpy()

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    prominence = 0.0125
    distance = 30
    sell_peaks, _ = find_peaks(sells, distance=distance, prominence=prominence)
    buy_peaks, _ = find_peaks(1 - sells, distance=distance, prominence=prominence)

    owns1 = np.zeros((sequence_length,))

    for peak in buy_peaks:
        sell_peak = sell_peaks[sell_peaks > peak]
        sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
        owns1[peak:sell_peak] = 1

    # TODO: handle the situation when there are 2 or more buys or sells in a row
    buys1 = owns1 == 1
    sells1 = owns1 == 0

    buys2 = mus > 0
    sells2 = mus < 0
    #buys2 = ha[:, 1] == ha[:, 3]
    #sells2 = ha[:, 1] == ha[:, 2]

    buys1 = buys1 & buys2
    sells1 = sells1 & sells2

    buys1 = buys1.astype(float)
    sells1 = sells1.astype(float)

    wealths, _, _, _, _ = get_wealths(
        X, buys1, sells1
    )

    print(wealths[-1] + 1)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks')
    ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks')
    #ax[0].plot(np.cumprod(mus + 1), alpha=0.7, label='smoothed price')
    ax[0].plot(wealths + 1, alpha=0.7, label='wealth')
    ax[0].legend()

    #ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    ax[1].plot(sell_peaks, sells[sell_peaks], 'ro', alpha=0.7, label='sell peaks')
    ax[1].plot(buy_peaks, sells[buy_peaks], 'go', alpha=0.7, label='buy peaks')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':

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

    sequence_length = 20000

    model = FFN(inputs, 1, use_lstm = True, Qlearn = False)
    model.load_state_dict(torch.load('models/' + model.name + '.pt'))
    model.eval()

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    plot_peaks(coin, files, inputs, params, model, sequence_length)
