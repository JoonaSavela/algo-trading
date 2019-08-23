import torch
from scipy.signal import find_peaks
import glob
from utils import *
from data import *
from model import *
import numpy as np
from optimize import get_wealths
import matplotlib.pyplot as plt

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def get_peaks(sells, prominence = 0.0125, distance = 30):
    sell_peaks, _ = find_peaks(sells, distance=distance, prominence=prominence)
    buy_peaks, _ = find_peaks(1 - sells, distance=distance, prominence=prominence)

    return buy_peaks, sell_peaks

def get_approx_peaks(sells, w = 15, th = 0.1):
    diffs = np.diff(sells)
    sells = sells[1:].reshape(-1, 1)
    sells = np.repeat(sells, 4, axis = 1)

    stoch = stochastic_oscillator(sells, w)
    diffs = diffs[-stoch.shape[0]:]
    idx = np.arange(stoch.shape[0])

    return idx[(stoch < th) & (diffs > 0)], idx[(stoch > 1 - th) & (diffs < 0)], stoch


def plot_peaks(files, inputs, params, model, sequence_length):
    X = load_all_data(files)

    alpha = 0.#35
    mus = smoothed_returns(X, alpha=alpha)
    mus = smoothed_returns(np.cumprod(mus + 1).reshape(-1, 1), alpha=alpha)

    obs, N, _ = get_obs_input(X, inputs, params)
    X = X[-N:, :]
    mus = mus[-N:]

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

    buy_peaks_approx, sell_peaks_approx, stoch = get_approx_peaks(sells)

    N = stoch.shape[0]

    X = X[-N:, :]
    mus = mus[-N:]
    sells = sells[-N:]

    buy_peaks, sell_peaks = get_peaks(sells)

    sequence_length = min(sequence_length, N)

    X = X[:sequence_length, :]
    mus = mus[:sequence_length]
    sells = sells[:sequence_length]
    stoch = stoch[:sequence_length]

    max_buy = buys.max()
    min_buy = buys.min()
    max_sell = sells.max()
    min_sell = sells.min()

    buys = (buys - min_buy) / (max_buy - min_buy)
    sells = (sells - min_sell) / (max_sell - min_sell)

    buy_peaks = buy_peaks[buy_peaks < sequence_length]
    sell_peaks = sell_peaks[sell_peaks < sequence_length]
    buy_peaks_approx = buy_peaks_approx[buy_peaks_approx < sequence_length]
    sell_peaks_approx = sell_peaks_approx[sell_peaks_approx < sequence_length]

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

    wealths1, _, _, _, _ = get_wealths(
        X, buys1, sells1
    )

    print(wealths1[-1] + 1)

    owns2 = np.zeros((sequence_length,))

    for peak in buy_peaks_approx:
        sell_peak = sell_peaks_approx[sell_peaks_approx > peak]
        sell_peak = sell_peak[0] if len(sell_peak) > 0 else sequence_length
        owns2[peak:sell_peak] = 1

    buys2 = owns2 == 1
    sells2 = owns2 == 0

    buys3 = mus > 0
    sells3 = mus < 0

    buys2 = buys2 & buys3
    sells2 = sells2 & sells3

    buys2 = buys2.astype(float)
    sells2 = sells2.astype(float)

    wealths2, _, _, _, _ = get_wealths(
        X, buys2, sells2
    )

    print(wealths2[-1] + 1)

    #lag = 15
    #threshold = 4
    #influence = 0.5
    #result = thresholding_algo(sells, lag, threshold, influence)
    #buys_approx = result['signals']
    #print(buys_approx.shape, buy_peaks.shape)

    plt.style.use('seaborn')
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].plot(X[:, 0] / X[0, 0], c='k', alpha=0.5, label='price')
    #ax[0].plot(sell_peaks, X[sell_peaks, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks')
    #ax[0].plot(buy_peaks, X[buy_peaks, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks')
    ax[0].plot(sell_peaks_approx, X[sell_peaks_approx, 0] / X[0, 0], 'ro', alpha=0.7, label='sell peaks approx')
    ax[0].plot(buy_peaks_approx, X[buy_peaks_approx, 0] / X[0, 0], 'go', alpha=0.7, label='buy peaks approx')
    #ax[0].plot(np.cumprod(mus + 1), alpha=0.7, label='smoothed price')
    ax[0].plot(wealths1 + 1, alpha=0.7, label='wealth1')
    ax[0].plot(wealths2 + 1, alpha=0.7, label='wealth2')
    ax[0].legend()

    #ax[1].plot(buys, c='g', alpha=0.5, label='buy')
    ax[1].plot(sells, c='r', alpha=0.5, label='sell')
    #ax[1].plot(sell_peaks, sells[sell_peaks], 'ro', alpha=0.7, label='sell peaks')
    #ax[1].plot(buy_peaks, sells[buy_peaks], 'go', alpha=0.7, label='buy peaks')
    ax[1].plot(buy_peaks_approx, sells[buy_peaks_approx], 'go', alpha=0.5, label='buy peaks approx')
    ax[1].plot(sell_peaks_approx, sells[sell_peaks_approx], 'ro', alpha=0.5, label='sell peaks approx')
    #ax[1].plot(stoch, 'b', alpha=0.5, label='signal')
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

    plot_peaks(files, inputs, params, model, sequence_length)
