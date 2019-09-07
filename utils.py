import numpy as np
import pandas as pd
from math import log10, floor
from scipy.signal import find_peaks
try:
    import torch
except ImportError as e:
    print(e)


def get_peaks(sells, prominence = 0.0125, distance = 30):
    sell_peaks, _ = find_peaks(sells, distance=distance, prominence=prominence)
    buy_peaks, _ = find_peaks(1 - sells, distance=distance, prominence=prominence)

    return buy_peaks, sell_peaks

def std_loss(out, sequence_length, batch_size, eps, e, n_epochs):
    starts = torch.randint(sequence_length // 2, (batch_size,))
    stds = []
    for b in range(batch_size):
        stds.append(out[starts[b]:starts[b]+sequence_length // 2, b, :2].std(dim = 0))
    return eps / (torch.stack(stds).mean() + eps) * 0.99 ** (e  / n_epochs)

def diff_loss(out, batch_size, use_tanh, e, n_epochs):
    diffs = []

    for b in range(batch_size):
        if use_tanh:
            sells = -out[:, b, 0].detach().numpy()
        else:
            sells = out[:, b, 1].detach().numpy()

        max_sell = sells.max()
        min_sell = sells.min()

        sells = (sells - min_sell) / (max_sell - min_sell)

        buy_peaks, sell_peaks = get_peaks(sells)
        buy_peaks = torch.from_numpy(buy_peaks)
        sell_peaks = torch.from_numpy(sell_peaks)

        if use_tanh:
            tmp1 = 1 - out[buy_peaks, b, 0]
            tmp2 = out[sell_peaks, b, 0] + 1

            # tmp1 = out[buy_peaks, b, 0].std()
            # tmp2 = out[sell_peaks, b, 0].std()
        else:
            tmp1 = out[buy_peaks, b, 1]
            tmp2 = 1 - out[sell_peaks, b, 1]

            # tmp1 = out[buy_peaks, b, 1].std()
            # tmp2 = out[sell_peaks, b, 1].std()

        diffs.append(tmp1)
        diffs.append(tmp2)

    return torch.cat(diffs).mean() * (1 - 0.99 ** (e / n_epochs))

def aggregate(X, n = 5):
    aggregated_X = np.zeros((X.shape[0] // n, X.shape[1]))
    X = X[-n * aggregated_X.shape[0]:, :]

    for i in range(aggregated_X.shape[0]):
        js = np.arange(i * n, (i + 1) * n)

        aggregated_X[i, 0] = X[js[-1], 0] # close
        aggregated_X[i, 1] = np.max(X[js, 1]) # high
        aggregated_X[i, 2] = np.min(X[js, 2]) # low
        aggregated_X[i, 3] = X[js[0], 3] # open

    return aggregated_X

def smoothed_returns(X, alpha = 0.75):
    returns = X[1:, 0] / X[:-1, 0] - 1

    mus = []
    mu_prior = 0.0

    for i in range(returns.shape[0]):
        if i > 0:
            mu_prior = mu_posterior

        mu_posterior = alpha * mu_prior + (1 - alpha) * returns[i]

        mus.append(mu_posterior)

    mus = np.array(mus)

    return mus

def std(X, window_size):
    return pd.Series(X[:, 0]).rolling(window_size).std().dropna().values

def sma(X, window_size):
    return np.convolve(X[:, 0], np.ones((window_size,))/window_size, mode='valid')

def stochastic_oscillator(X, window_size = 3 * 14, k = 1, latency = 0):
    res = []
    for i in range(X.shape[0] - window_size + 1 - latency):
        max_price = np.max(X[i:i + window_size, 1])
        min_price = np.min(X[i:i + window_size, 2])
        min_close = np.min(X[i:i + window_size, 0])
        stoch = (X[i + window_size - 1, 0] - min_close) / (max_price - min_price)
        res.append(stoch)
    res = np.array(res)
    res = np.convolve(res, np.ones((k,))/k, mode='valid')
    return res

def heikin_ashi(X):
    res = np.zeros((X.shape[0] - 1, 4))
    for i in range(X.shape[0] - 1):
        ha_close = 0.25 * np.sum(X[i + 1, :4])
        if i == 0:
            ha_open = 0.5 * (X[0, 0] + X[0, 3])
        else:
            ha_open = 0.5 * np.sum(res[i - 1, :2])
        ha_high = np.max([X[i + 1, 1], ha_close, ha_open])
        ha_low = np.min([X[i + 1, 2], ha_close, ha_open])
        res[i, :] = [ha_close, ha_open, ha_high, ha_low]
    return res


def round_to_n(x, n = 2):
    if x == 0: return x
    res = round(x, -int(floor(log10(abs(x)))) + (n - 1)) if x != 0 else 0
    res = int(res) if abs(res) >= 10**(n - 1) else res
    return res

def floor_to_n(x, n = 2):
    if x == 0: return x
    p = -int(floor(log10(abs(x)))) + (n - 1)
    res = floor(x * 10 ** p) / 10 ** p
    res = int(res) if abs(res) >= 10**(n - 1) else res
    return res


def get_time(filename):
    split1 = filename.split('/')
    split2 = split1[2].split('.')
    return int(split2[0])


def get_obs_input(X, inputs, params):
    obs = []

    if 'ma' not in inputs:
        raise ValueError('"ma" must be included in the parameters in order to normalize the prices')
    else:
        ma_window_min_max = params['ma_window_min_max']
        if inputs['ma'] == 0:
            window_sizes = ma_window_min_max[1:]
        else:
            window_sizes = np.round(
                np.linspace(ma_window_min_max[0], ma_window_min_max[1], inputs['ma'] + 1)
            ).astype(int)

        tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
        ma_ref = sma(tp, window_sizes[-1])

        N = ma_ref.shape[0]

        for w in window_sizes[:-1]:
            ma = sma(tp, w)[-N:] / ma_ref
            obs.append(ma)

    if 'price' in inputs:
        prices = X[-N:, :inputs['price']]
        tmp = np.split(prices, inputs['price'], axis = 1)
        for i in range(len(tmp)):
            tmp[i] = tmp[i].reshape(-1) / ma_ref
        obs.extend(tmp)

    if 'mus' in inputs:
        returns = X[1:, 0] / X[:-1, 0] - 1

        r = returns

        for i in range(inputs['mus']):
            r = smoothed_returns(np.cumprod(r + 1).reshape(-1, 1), alpha = params['alpha'])
            obs.append(r[-N:])

    if 'std' in inputs:
        std_window_min_max = params['std_window_min_max']
        window_sizes = np.round(
            np.linspace(std_window_min_max[0], std_window_min_max[1], inputs['std'])
        ).astype(int)

        for w in window_sizes:
            sd = std(tp, w)[-N:] / ma_ref
            obs.append(sd)

    if 'ha' in inputs:
        ha = heikin_ashi(X)[-N:, :inputs['ha']]
        tmp = np.split(ha, inputs['ha'], axis = 1)
        for i in range(len(tmp)):
            tmp[i] = tmp[i].reshape(-1) / ma_ref
        obs.extend(tmp)

    if 'stoch' in inputs:
        stoch_window_min_max = params['stoch_window_min_max']
        window_sizes = np.round(
            np.linspace(stoch_window_min_max[0], stoch_window_min_max[1], inputs['stoch'])
        ).astype(int)

        for w in window_sizes:
            stoch = stochastic_oscillator(X, w)[-N:]
            obs.append(stoch)


    obs = np.stack(obs, axis = 1)
    obs = torch.from_numpy(obs).type(torch.float32)

    ma_ref = torch.from_numpy(ma_ref).type(torch.float32)

    #print(obs.shape)

    return obs, N, ma_ref

def init_state(inputs, batch_size, initial_usd = 1.0, initial_coin = 0.0):
    state = []

    capital_usd = initial_usd
    state.append(capital_usd)

    capital_coin = initial_coin
    state.append(capital_coin)

    timedelta = -1.0
    state.append(timedelta)

    buy_price = -1.0
    state.append(buy_price)

    state = torch.tensor(state).view(1, -1).repeat(batch_size, 1).type(torch.float32)

    return state
