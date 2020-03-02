import numpy as np
import pandas as pd
from math import log10, floor, ceil
from scipy.signal import find_peaks
from scipy import stats
try:
    import matplotlib.pyplot as plt
    import torch
except ImportError as e:
    print(e)


def rolling_max(X, w):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return pd.Series(X[:, 0]).rolling(w).max().dropna().values


def rolling_min(X, w):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return pd.Series(X[:, 0]).rolling(w).min().dropna().values


def p_to_logit(p):
    return np.log(p / (1 - p))

def logit_to_p(logit):
    return 1 / (1 + np.exp(-logit))

def rolling_quantile(X, w):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return pd.Series(X[:, 0]).rolling(w).apply(lambda x: stats.percentileofscore(x, x[-1]), raw=True).dropna().values * 0.01

def apply_trend(x, limits, trend, strength):
    min_x = limits[0]
    range_x = limits[1] - limits[0]
    p = (x - min_x) / range_x

    logit = p_to_logit(p) + trend * strength
    p = logit_to_p(logit)

    x = p * range_x + min_x
    return x


def risk_management(X, trends, risk_args, limits, commissions = 0.00075):
    # base_stop_loss = np.log(1 - risk_args['base_stop_loss'])
    base_stop_loss = risk_args['base_stop_loss']
    profit_fraction = risk_args['profit_fraction']
    increment = np.log(1 + risk_args['increment'])
    trailing_alpha = risk_args['trailing_alpha']
    steam_th = risk_args['steam_th']
    strength = risk_args['strength']

    i = 0
    N = X.shape[0]

    buy_price = None
    buy_time = None
    sell_price = X[0, 0]
    sell_time = 0
    trailing_level = 0.0

    risk_buys = np.zeros((N,)).astype(bool)
    risk_sells = np.zeros((N,)).astype(bool)

    trades = []

    stop_losses = []

    # TODO: reduce repetition
    # TODO: return information about the pseudo trades
    # TODO: use trend to change thresholds
    while i < N:
        no_position = buy_price is None and buy_time is None and \
                        sell_price is not None and sell_time is not None

        if no_position:
            log_prices = - np.log(X[sell_time:i + 1, 0] / sell_price)
            trend = - trends[i]
        else:
            log_prices = np.log(X[buy_time:i + 1, 0] / buy_price)
            trend = trends[i]

        log_return = log_prices[-1]
        max_log_return = np.max(log_prices)

        trailing_n = np.floor(max_log_return / increment).astype(int)
        trailing_level = trailing_n * increment
        event_time = np.argmax(log_prices > trailing_level)

        if no_position:
            event_time += sell_time
        else:
            event_time += buy_time

        stop_loss = apply_trend(base_stop_loss, limits['base_stop_loss'], trend, strength) # TODO: add trend
        stop_losses.append(stop_loss)
        stop_loss = np.log(1 - stop_loss)
        take_profit = - profit_fraction * stop_loss
        stop_loss = trailing_level - \
                        increment * trailing_alpha * (1 - trailing_alpha ** trailing_n) / \
                        (1 - trailing_alpha) + \
                        stop_loss * trailing_alpha ** trailing_n





        if no_position:
            if log_return < stop_loss:
                risk_buys[i] = True
            elif (i - event_time) * (log_return - trailing_level) < steam_th:
                risk_buys[i] = True


            if risk_buys[i]:
                buy_price = X[i, 0]
                buy_time = i
                sell_price = None
                sell_time = None


        else:
            min_i = np.argmin(log_prices)
            cause = '–'

            if log_return < stop_loss:
                risk_sells[i] = True
                cause = 'stop loss'
            elif np.log(X[i, 0] / X[buy_time + min_i, 0]) > take_profit:
                risk_sells[i] = True
                cause = 'take profit'
            elif (i - event_time) * (log_return - trailing_level) < steam_th:
                risk_sells[i] = True
                cause = 'steam'

            if risk_sells[i]:
                sell_price = X[i, 0]
                sell_time = i

                trade = {
                    'buy_time': buy_time,
                    'sell_time': sell_time,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'cause': cause,
                }

                trades.append(trade)

                buy_price = None
                buy_time = None


        i += 1

    # plt.hist(stop_losses, 50)
    # plt.show()

    trades = pd.DataFrame(trades)

    trades['profit'] = np.log(trades['sell_price'] / trades['buy_price']) + np.log(1 - commissions) * 2
    trades['winning'] = trades['profit'] > 0

    return risk_buys, risk_sells, trades


def get_ad(X, w, cumulative = True):
    lows = pd.Series(X[:, 0]).rolling(w).min().dropna().values
    highs = pd.Series(X[:, 0]).rolling(w).max().dropna().values

    N = lows.shape[0]

    cmfv = ((X[-N:, 0] - lows) - (highs - X[-N:, 0])) / (highs - lows) * X[-N:, 5]

    if cumulative:
        return np.cumsum(cmfv)

    return cmfv

def get_obv(X, cumulative = True, use_sign = True):
    log_returns = np.log(X[1:, 0] / X[:-1, 0])

    if use_sign:
        log_returns = np.sign(log_returns)

    signed_volumes = X[1:, 5] * log_returns

    if cumulative:
        return np.cumsum(signed_volumes)

    return signed_volumes


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

def ema(X, alpha, mu_prior = 0.0):
    alpha_is_not_float = not (type(alpha) is float or type(alpha) is np.float64)
    if alpha_is_not_float:
        alphas = alpha

    if len(X.shape) > 1:
        X = X[:, 0]

    mus = []

    for i in range(X.shape[0]):
        if i > 0:
            mu_prior = mu_posterior

        if alpha_is_not_float:
            alpha = alphas[i]
        mu_posterior = alpha * mu_prior + (1 - alpha) * X[i]

        mus.append(mu_posterior)

    mus = np.array(mus)

    return mus


def smoothed_returns(X, alpha = 0.75, n = 1):
    # returns = X[1:, 0] / X[:-1, 0] - 1
    returns = np.log(X[1:, 0] / X[:-1, 0])

    for i in range(n):
        returns = ema(returns, alpha = alpha)

    return returns

def std(X, window_size):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return pd.Series(X[:, 0]).rolling(window_size).std().dropna().values

def sma(X, window_size):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return np.convolve(X[:, 0], np.ones((window_size,))/window_size, mode='valid')

def stochastic_oscillator(X, window_size = 3 * 14, k = 1, latency = 0):
    if len(X.shape) == 1:
        X = np.repeat(X.reshape((-1, 1)), 4, axis = 1)
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

def ceil_to_n(x, n = 2):
    if x == 0: return x
    p = -int(floor(log10(abs(x)))) + (n - 1)
    res = ceil(x * 10 ** p) / 10 ** p
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
            tmp[i] = tmp[i].reshape(-1) / (ma_ref if i < 4 else 1)
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
