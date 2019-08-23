import torch
from scipy.signal import find_peaks
import glob
from utils import *
from data import *
from model import *
import numpy as np
import matplotlib.pyplot as plt

def plot_peaks(coin, files, inputs, params, model, sequence_length):
    X = load_all_data(files)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]

    X = X[:sequence_length, :]
    obs = obs[:sequence_length, :]

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

    sell_peaks, _ = find_peaks(sells, distance=15, prominence=0.05)
    buy_peaks, _ = find_peaks(1 - sells, distance=15, prominence=0.05)
    print(buy_peaks[-1])

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks')
    ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks')
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

    sequence_length = 700

    model = FFN(inputs, 1, use_lstm = True, Qlearn = False)
    model.load_state_dict(torch.load('models/' + model.name + '.pt'))
    model.eval()

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    plot_peaks(coin, files, inputs, params, model, sequence_length)
