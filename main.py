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
from peaks import get_peaks

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([0, 1, 2, 3])
    # idx = np.arange(2)
    test_files = np.array(test_files)[-2:]
    X = load_all_data(test_files)
    # X = X[:1000, :]
    # X = X[:1200, :]

    # min alpha: 0.9
    alpha = 0.975

    emaX = ema(X, alpha, mu_prior = X[0, 0])

    # tmp = np.log(X[:, 0] / emaX)
    tmp = X[:, 0] / emaX
    stoch = stochastic_oscillator(tmp, 500)

    th = 0.2
    th1 = -np.log((1 - th) / th)
    # print(th1)

    # min alpha1: 0.99
    alpha1 = 0.995
    emaX1 = ema(X, alpha1, mu_prior = X[0, 0])
    _emaX = emaX1[1:] / emaX1[:-1]

    # plt.hist(_emaX)
    # plt.show()

    _emaX = np.log(_emaX)
    _emaX_std = std(_emaX, 500)

    N = _emaX_std.shape[0]
    _emaX = _emaX[-N:]

    _emaX /= _emaX_std + 10 ** (-5)
    # _emaX = np.exp(_emaX)

    # _emaX -= 1
    # _emaX /= np.std(_emaX)
    # _emaX += 1

    # _emaX = () ** 2000

    buy_ths = 1 / (1 + np.exp(-(th1 + _emaX)))
    sell_ths = 1 / (1 + np.exp(-(-th1 + _emaX)))

    # plt.hist(_emaX)
    # plt.show()
    #
    # plt.hist(buy_ths)
    # plt.show()
    #
    # plt.hist(sell_ths)
    # plt.show()

    # print(buy_ths)

    N = min(stoch.shape[0], N)
    X = X[-N:, :]
    emaX = emaX[-N:]
    emaX1 = emaX1[-N:]
    buy_ths = buy_ths[-N:]
    sell_ths = sell_ths[-N:]

    fig, ax = plt.subplots(ncols = 2)

    ax[0].plot(X[:, 0])
    ax[0].plot(emaX)
    ax[0].plot(emaX1)

    ax[1].plot(stoch)
    ax[1].plot(buy_ths)
    ax[1].plot(sell_ths)

    plt.show()

    # cumulative = True
    # signed_volumes1 = get_obv(X, cumulative, True)
    # signed_volumes2 = get_obv(X, cumulative, False)
    #
    # # fig, ax = plt.subplots(ncols = 3)
    #
    # # ax[0].plot(X[1:, 0])
    # # ax[1].plot(signed_volumes1)
    # # ax[2].plot(signed_volumes2)
    # # plt.show()
    #
    # cmfv = get_ad(X, 300, False)
    #
    # mus = smoothed_returns(X, alpha = 0.7, n = 4)
    # smoothed_X = np.exp(np.cumsum(mus))
    #
    # w = 15
    # mins = pd.Series(smoothed_X).rolling(w).min().dropna().values
    # maxs = pd.Series(smoothed_X).rolling(w).max().dropna().values
    # ranges = np.log(maxs / mins)
    #
    # # stds = std(smoothed_X, 5) ** 0.5
    # # stds = std(X, 5) ** 0.5
    # stds = std(mus, 30) ** 0.5
    # # print(stds.min())
    # # print(stds)
    #
    # N = min(cmfv.shape[0], smoothed_X.shape[0], stds.shape[0], ranges.shape[0])
    #
    # modified_mus = mus[-N:] / (stds[-N:] + 0.025)
    # modified_smoothed_X = np.exp(np.cumsum(modified_mus))
    #
    # commissions = 0.0015
    #
    # th = 0.01
    # buys1 = modified_mus > th
    # sells1 = modified_mus <= 0
    #
    # buys1 = buys1[-N:]
    # sells1 = sells1[-N:]
    #
    # buys2 = ranges > np.log(1 + commissions)
    # sells2 = buys2
    #
    # buys2 = buys2[-N:]
    # sells2 = sells2[-N:]
    #
    # buys3 = cmfv > 0
    # sells3 = cmfv <= 0
    #
    # buys3 = buys3[-N:]
    # sells3 = sells3[-N:]
    #
    # # buys = buys1 & buys2 & buys3
    # # sells = sells1 & sells2 & sells3
    #
    # buys = buys1 & buys3
    # sells = sells1 & sells3
    #
    # initial_usd = 1000
    #
    # wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
    #     X[-N:, :], buys, sells, initial_usd = initial_usd, #commissions = 0
    # )
    #
    # print(wealths[-1] + 1)
    #
    # fig, ax = plt.subplots(ncols = 2, nrows = 2)
    #
    # ax[0, 0].plot(X[-N:, 0] / X[0, 0], c='k', alpha=0.5)
    # ax[0, 0].plot(smoothed_X[-N:], c='b', alpha=0.7)
    # ax[0, 0].plot(wealths[-N:] + 1, alpha=0.7)
    # ax[0, 1].plot(np.cumsum(cmfv[-N:]))
    # ax[1, 0].plot(stds[-N:])
    # ax[1, 0].plot(ranges[-N:])
    # ax[1, 0].axhline(np.log(1 + commissions))
    # # ax[1, 0].plot(modified_mus[-N:])
    # ax[1, 1].plot(modified_smoothed_X[-N:])
    # ax[1, 1].plot(smoothed_X[-N:] ** 6)
    # plt.show()




    # inputs = {
    #     'price': 4,
    #     'mus': 3,
    #     'std': 3,
    #     'ma': 3,
    #     'ha': 4,
    #     'stoch': 3,
    # }

    # inputs = {
    #     'price': 4,
    #     'mus': 4,
    #     'std': 4,
    #     'ma': 4,
    #     'ha': 4,
    #     'stoch': 4,
    # }
    # hidden_size = sum(list(inputs.values())) * 1
    #
    # N = 1500
    #
    # params = {
    #     'alpha': 0.8,
    #     'std_window_min_max': [30, N],
    #     'ma_window_min_max': [30, N],
    #     'stoch_window_min_max': [30, N],
    # }
    #
    # sequence_length = 800000
    #
    # signal_model = FFN(inputs, 1, use_lstm = True, Qlearn = False, use_tanh = False, hidden_size = hidden_size)
    # signal_model.load_state_dict(torch.load('models/' + signal_model.name + '.pt'))
    # signal_model.eval()
    #
    # model = FFN(dict(n_inputs=3), 1, use_tanh = True, hidden_size = 16)
    # model.load_state_dict(torch.load('models/' + model.name + '.pt'))
    # model.eval()
    #
    # alpha = 0.#5
    # mus = smoothed_returns(X, alpha=alpha)
    # mus = smoothed_returns(np.cumprod(mus + 1).reshape(-1, 1), alpha=alpha)
    #
    # obs, N, _ = get_obs_input(X, inputs, params)
    # X = X[-N:, :]
    # mus = mus[-N:]
    #
    # sequence_length = min(sequence_length, N)
    #
    # X = X[:sequence_length, :]
    # obs = obs[:sequence_length, :]
    # mus = mus[:sequence_length]
    #
    # signal_model.init_state()
    # model.init_state()
    #
    # inp = obs.unsqueeze(1)
    # signal_out = signal_model(inp)
    #
    # buy_peaks, sell_peaks = get_peaks(signal_out[:, 0, 1].detach().numpy())
    #
    # out = model(signal_out).detach().numpy()
    #
    # buys = np.zeros(sequence_length)
    # sells = np.zeros(sequence_length)
    #
    # li = out[:, 0, 0] > 0
    # buys[li] = out[li, 0, 0]
    #
    # li = out[:, 0, 0] < 0
    # sells[li] = -out[li, 0, 0]
    #
    # max_buy = buys.max()
    # min_buy = buys.min()
    # max_sell = sells.max()
    # min_sell = sells.min()
    #
    # buys = (buys - min_buy) / (max_buy - min_buy)
    # sells = (sells - min_sell) / (max_sell - min_sell)
    #
    # # buys = buys ** 2
    # # sells = sells ** 2
    #
    # idx = np.arange(1, sequence_length)
    #
    # th = 0.5
    # buys_li = (buys[1:] > th) & (np.diff(buys) < 0)
    # buy_peaks_approx = idx[buys_li]
    #
    # sells_li = (sells[1:] > th) & (np.diff(sells) < 0)
    # sell_peaks_approx = idx[sells_li]
    #
    # d = 0
    # buy_peaks_approx += d
    # sell_peaks_approx += d
    # buy_peaks_approx = buy_peaks_approx[buy_peaks_approx < sequence_length]
    # sell_peaks_approx = sell_peaks_approx[sell_peaks_approx < sequence_length]
    #
    # owns1 = np.zeros((sequence_length,))
    #
    # for peak in buy_peaks_approx:
    #     sell_peak = sell_peaks_approx[sell_peaks_approx > peak]
    #     sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
    #     owns1[peak:sell_peak] = 1
    #
    # buys1 = owns1 == 1
    # sells1 = owns1 == 0
    #
    # buys2 = mus > 0
    # sells2 = mus < 0
    #
    # buys1 = buys1 & buys2
    # sells1 = sells1 & sells2
    #
    # buys1 = buys1.astype(float)
    # sells1 = sells1.astype(float)
    #
    # initial_usd = 1000
    #
    # wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
    #     X, buys1, sells1, initial_usd = initial_usd,
    # )
    #
    # print(wealths[-1] + 1)
    #
    # plt.style.use('seaborn')
    # fig, ax = plt.subplots(ncols=2, figsize=(16, 8))
    #
    # ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    # ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'mo', ms=12.0, alpha=0.7, label='sell peaks')
    # ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'co', ms=12.0, alpha=0.7, label='buy peaks')
    # ax[0].plot(sell_peaks_approx, X[sell_peaks_approx, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks approx')
    # ax[0].plot(buy_peaks_approx, X[buy_peaks_approx, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks approx')
    # ax[0].plot(wealths + 1, c='b', alpha = 0.5, label='wealth')
    # ax[0].legend()
    #
    # signal_sells = signal_out[:, 0, 1].detach().numpy()
    # ax[1].plot(signal_sells, c='k', alpha=0.5, label='signal')
    # ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    # ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    # ax[1].plot(sell_peaks, signal_sells[sell_peaks], 'mv', alpha=0.7, label='sell peaks')
    # ax[1].plot(buy_peaks, signal_sells[buy_peaks], 'c^', alpha=0.7, label='buy peaks')
    # ax[1].plot(sell_peaks_approx, sells[sell_peaks_approx], 'ro', alpha=0.7, label='sell peaks approx')
    # ax[1].plot(buy_peaks_approx, buys[buy_peaks_approx], 'go', alpha=0.7, label='buy peaks approx')
    # ax[1].legend()
    # plt.show()


if __name__ == "__main__":
    main()
