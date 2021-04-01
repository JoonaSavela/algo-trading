if __name__ == "__main__":
    raise ValueError("algtra/utils.py should not be run as main.")

import numpy as np
import pandas as pd
import json
from math import log10, floor, ceil
from scipy import stats
from itertools import product
import os
import sys
import time
from algtra.constants import TAX_RATE, PARTIAL_LEVERAGED_SYMBOLS

# matplotlib is not installed in cloud since it is not needed
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)


def seconds_to_milliseconds(t, units):
    if t < 1:
        t *= 1000
        units = "milliseconds"

    return t, units


def milliseconds_to_microseconds(t, units):
    if t < 1:
        t *= 1000
        units = "microseconds"

    return t, units


def print_times(title, subtitles, times):
    assert len(subtitles) == len(times)

    max_subtitle_length = max(len(subtitle) for subtitle in subtitles)

    print(f"{title}:")

    for i, t in enumerate(times):
        units = "seconds"
        t, units = seconds_to_milliseconds(t, units)
        t, units = milliseconds_to_microseconds(t, units)
        t = round_to_n(t)

        n_tabs = (max_subtitle_length - len(subtitles[i])) // 4 + 1
        print(f"\t{subtitles[i]}:" + "\t" * n_tabs, f"{t} {units}")

    print()


def append_to_dict_of_collections(d, k, v, collection_type="list"):
    if k not in d:
        if collection_type == "list":
            d[k] = []
        elif collection_type == "deque":
            d[k] = deque()
        else:
            raise ValueError("Invalid collection_type")

    d[k].append(v)

    return d


def get_coin(symbol):
    for partial_leveraged_symbol in PARTIAL_LEVERAGED_SYMBOLS:
        symbol = symbol.replace(partial_leveraged_symbol, "")

    if not symbol:
        coin = "BTC"
    else:
        coin = symbol

    return coin


def get_symbol(coin, m, bear=False):
    res = coin
    if m > 1 or bear:
        if coin == "BTC":
            res = ""
        if bear:
            res += "BEAR" if m > 1 else "HEDGE"
        else:
            res += "BULL"

    return res


def apply_taxes(trade_wealths, copy=False):
    if isinstance(trade_wealths, float):
        if trade_wealths > 1:
            trade_wealths = (trade_wealths - 1) * (1 - TAX_RATE) + 1
    else:
        if copy:
            trade_wealths = np.copy(trade_wealths)
        li = trade_wealths > 1
        trade_wealths[li] = (trade_wealths[li] - 1) * (1 - TAX_RATE) + 1

    return trade_wealths


def get_total_balance(client, separate=True, filter=None):
    balances = pd.DataFrame(client.get_balances()).set_index("coin")
    time.sleep(0.05)
    if filter is not None:
        li = [ix in filter for ix in balances.index]
        balances = balances.loc[li, :]
    total_balance = balances["usdValue"].sum()

    if separate:
        return total_balance, balances

    return total_balance


def get_average_trading_period(strategies, unique=True):
    aggregate_Ns = np.array([v["params"][0] * 60 for v in strategies.values()])
    if unique:
        aggregate_Ns = np.unique(aggregate_Ns)
    max_trading_frequencies_per_min = 1 / aggregate_Ns
    total_max_trading_frequencies_per_min = np.sum(max_trading_frequencies_per_min)

    return int(1 / total_max_trading_frequencies_per_min)


def get_parameter_names(strategy_type):
    if strategy_type == "ma":
        return ["aggregate_N", "w"]
    elif strategy_type == "stoch":
        return ["aggregate_N", "w", "th"]
    elif strategy_type == "macross":
        return ["aggregate_N", "w", "w2"]
    else:
        raise ValueError("strategy_type")


def choose_get_buys_and_sells_fn(strategy_type):
    if strategy_type == "ma":
        return get_buys_and_sells_ma
    elif strategy_type == "stoch":
        return get_buys_and_sells_stoch
    elif strategy_type == "macross":
        return get_buys_and_sells_macross
    else:
        raise ValueError("strategy_type")


def get_filtered_strategies_and_weights(
    coins=["ETH", "BTC"],
    freqs=["low", "high"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    normalize=True,
    filter=True,
):

    strategies = {}

    for coin, freq, strategy_type, m, m_bear in product(
        coins, freqs, strategy_types, ms, m_bears
    ):
        strategy_key = "_".join([coin, freq, strategy_type, str(m), str(m_bear)])
        filename = f"optim_results/{strategy_key}.json"
        if os.path.exists(filename):
            with open(filename, "r") as file:
                strategies[strategy_key] = json.load(file)

    if os.path.exists("optim_results/weights.json") and filter:
        with open("optim_results/weights.json", "r") as file:
            weights = json.load(file)

        filtered_strategies = {}
        filtered_weights = {}

        for k in strategies.keys():
            if k in weights and weights[k] > 0:
                filtered_strategies[k] = strategies[k]
                filtered_weights[k] = weights[k]

        strategies = filtered_strategies
        weights = filtered_weights
    else:
        weights = {}

    if len(weights) > 0:
        if normalize:
            weight_values = np.array(list(weights.values()))
            weight_values /= np.sum(weight_values)
            weights = dict(list(zip(weights.keys(), weight_values)))
    else:
        weights = dict(
            list(zip(strategies.keys(), np.ones((len(strategies),)) / len(strategies)))
        )

    return strategies, weights


# This function computes GCD (Greatest Common Divisor)
def get_gcd(x, y):
    while y:
        x, y = y, x % y
    return x


# This function computes LCM (Least Common Multiple)
def get_lcm(x, y):
    return x * y // get_gcd(x, y)


def quick_plot(xs, colors, alphas, log=False):
    plt.style.use("seaborn")
    for i in range(len(xs)):
        plt.plot(xs[i], color=colors[i], alpha=alphas[i])
    if log:
        plt.yscale("log")
    plt.show()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)


def print_dict(d, pad=""):
    for k, v in d.items():
        print(pad, k)
        if isinstance(v, dict):
            print_dict(v, pad + "\t")
        else:
            print(pad + " ", v)


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
    return (
        pd.Series(X[:, 0])
        .rolling(w)
        .apply(lambda x: stats.percentileofscore(x, x[-1]), raw=True)
        .dropna()
        .values
        * 0.01
    )


def apply_trend(x, limits, trend, strength):
    min_x = limits[0]
    range_x = limits[1] - limits[0]
    p = (x - min_x) / range_x

    logit = p_to_logit(p) + trend * strength
    p = logit_to_p(logit)

    x = p * range_x + min_x
    return x


def get_ad(X, w, cumulative=True):
    lows = pd.Series(X[:, 0]).rolling(w).min().dropna().values
    highs = pd.Series(X[:, 0]).rolling(w).max().dropna().values

    N = lows.shape[0]

    cmfv = ((X[-N:, 0] - lows) - (highs - X[-N:, 0])) / (highs - lows) * X[-N:, 5]

    if cumulative:
        return np.cumsum(cmfv)

    return cmfv


def get_obv(X, cumulative=True, use_sign=True):
    log_returns = np.log(X[1:, 0] / X[:-1, 0])

    if use_sign:
        log_returns = np.sign(log_returns)

    signed_volumes = X[1:, 5] * log_returns

    if cumulative:
        return np.cumsum(signed_volumes)

    return signed_volumes


def aggregate(X, n=5, from_minute=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if from_minute:
        n *= 60

    aggregated_X = np.zeros((X.shape[0] // n, X.shape[1]))

    idx = np.arange((X.shape[0] % n) + n - 1, X.shape[0], n)
    aggregated_X[:, 0] = X[idx, 0]  # close

    if X.shape[1] >= 2:
        highs = rolling_max(X[:, 1], n)
        idx = np.arange(X.shape[0] % n, len(highs), n)
        aggregated_X[:, 1] = highs[idx]  # high

    if X.shape[1] >= 3:
        lows = rolling_min(X[:, 2], n)
        idx = np.arange(X.shape[0] % n, len(lows), n)
        aggregated_X[:, 2] = lows[idx]  # low

    if X.shape[1] >= 4:
        idx = np.arange(X.shape[0] % n, X.shape[0], n)
        aggregated_X[:, 3] = X[idx, 3]  # open

    if X.shape[1] > 4:
        X = X[-n * aggregated_X.shape[0] :, :]

        for i in range(aggregated_X.shape[0]):
            js = np.arange(i * n, (i + 1) * n)

            # aggregated_X[i, 1] = np.max(X[js, 1])
            # aggregated_X[i, 2] = np.min(X[js, 2])
            aggregated_X[i, 4] = np.sum(X[js, 4])
            aggregated_X[i, 5] = np.sum(X[js, 5])

    return aggregated_X


def ema(X, alpha, mu_prior=0.0):
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


def smoothed_returns(X, alpha=0.75, n=1):
    # returns = X[1:, 0] / X[:-1, 0] - 1
    returns = np.log(X[1:, 0] / X[:-1, 0])

    for i in range(n):
        returns = ema(returns, alpha=alpha)

    return returns


def std(X, window_size):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return pd.Series(X[:, 0]).rolling(window_size).std().dropna().values


def sma_old(X, window_size):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    return np.convolve(X[:, 0], np.ones((window_size,)) / window_size, mode="valid")


def sma(X, window_size):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    S = np.sum(X[:window_size, 0])
    res = [S]

    for i in range(len(X) - window_size):
        S += X[i + window_size, 0] - X[i, 0]
        res.append(S)

    res = np.array(res) / window_size

    # assert(np.allclose(res, sma_old(X, window_size)))

    return res


# TODO: make this faster with concurrent.futures!
def stochastic_oscillator(X, window_size=3 * 14, k=1):
    if len(X.shape) == 1:
        X = np.repeat(X.reshape((-1, 1)), 4, axis=1)
    res = []
    for i in range(X.shape[0] - window_size + 1):
        max_price = np.max(X[i : i + window_size, 1])
        min_price = np.min(X[i : i + window_size, 2])
        min_close = np.min(X[i : i + window_size, 0])
        stoch = (X[i + window_size - 1, 0] - min_close) / (max_price - min_price)
        res.append(stoch)
    res = np.array(res)
    if k > 1:
        res = np.convolve(res, np.ones((k,)) / k, mode="valid")
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


def round_to_n(x, n=3):
    if x == 0:
        return x
    res = round(x, -int(floor(log10(abs(x)))) + (n - 1)) if x != 0 else 0
    res = int(res) if abs(res) >= 10 ** (n - 1) else res
    return res


def floor_to_n(x, n=3):
    if x == 0:
        return x
    p = -int(floor(log10(abs(x)))) + (n - 1)
    res = floor(x * 10 ** p) / 10 ** p
    res = int(res) if abs(res) >= 10 ** (n - 1) else res
    return res


def ceil_to_n(x, n=3):
    if x == 0:
        return x
    p = -int(floor(log10(abs(x)))) + (n - 1)
    res = ceil(x * 10 ** p) / 10 ** p
    res = int(res) if abs(res) >= 10 ** (n - 1) else res
    return res


def get_time(filename):
    split1 = filename.split("/")
    split2 = split1[2].split(".")
    return int(split2[0])


# TODO: test that with m = 1, X doesn't change
def get_multiplied_X(X, multiplier=1):
    if X.shape[1] > 1:
        returns = X[1:, 3] / X[:-1, 3] - 1
        returns = multiplier * returns
        assert np.all(returns > -1.0)

        X_res = np.zeros_like(X)
        X_res[:, 3] = np.concatenate([[1.0], np.cumprod(returns + 1)])

        other_returns = X[:, :3] / X[:, 3].reshape((-1, 1)) - 1
        other_returns = multiplier * other_returns
        assert np.all(other_returns > -1.0)
        X_res[:, :3] = X_res[:, 3].reshape((-1, 1)) * (other_returns + 1)

        if multiplier < 0:
            X_res[:, [1, 2]] = X_res[:, [2, 1]]

        X_res[:, 4:] = X[:, 4:]
    else:
        returns = X[1:, 0] / X[:-1, 0] - 1
        returns = multiplier * returns
        assert np.all(returns > -1.0)

        X_res = np.zeros_like(X)
        X_res[:, 0] = np.concatenate([[1.0], np.cumprod(returns + 1)])

    return X_res


def get_max_dropdown(wealths, return_indices=False):
    res = np.Inf
    I = -1
    J = -1

    for i in range(len(wealths)):
        j = np.argmin(wealths[i:]) + i
        dropdown = wealths[j] / wealths[i]
        if dropdown < res:
            res = dropdown
            I = i
            J = j

    if return_indices:
        return res, I, J

    return res


def get_or_create(d, k, create_func, *args):
    if not k in d:
        d[k] = create_func(k, *args)

    return d[k]


def get_entry_and_exit_idx(entries, exits, N):
    idx = np.arange(N)

    entries_li = np.diff(np.concatenate((np.array([0]), entries))) == 1
    entries_idx = idx[entries_li]

    exits_li = np.diff(np.concatenate((np.array([0]), exits))) == 1
    exits_idx = idx[exits_li]

    return entries_idx, exits_idx


def get_buys_and_sells_ma(X, aggregate_N, w, as_boolean=False, from_minute=True):
    w = aggregate_N * w
    if from_minute:
        w *= 60

    diff = X[w:, 0] - X[:-w, 0]
    N = diff.shape[0]

    # Calculating buys and sells from "diff" is equivalent to
    #   1. calculating a sma from X
    #   2. calculating buys and sells from the difference of this sma
    buys = diff > 0
    sells = diff < 0

    return_tuple = (N,)

    if as_boolean:
        if N == 1:
            return (buys[0], sells[0]) + return_tuple
        else:
            return (buys, sells) + return_tuple

    buys = buys.astype(float)
    sells = sells.astype(float)

    return (buys, sells) + return_tuple


def get_buys_and_sells_macross(
    X, aggregate_N, w_max, w_min, as_boolean=False, from_minute=True
):
    w_max = aggregate_N * w_max
    w_min = aggregate_N * w_min
    if from_minute:
        w_max *= 60
        w_min *= 60

    assert w_max > w_min

    ma_max = sma(X[1:, 0] / X[0, 0], w_max)
    ma_min = sma(X[(w_max - w_min + 1) :, 0] / X[0, 0], w_min)

    N = ma_max.shape[0]
    assert N == len(X) - w_max
    assert N == ma_min.shape[0]

    buys = ma_min > ma_max
    sells = ma_min < ma_max

    return_tuple = (N,)

    if as_boolean:
        if N == 1:
            return (buys[0], sells[0]) + return_tuple
        else:
            return (buys, sells) + return_tuple

    buys = buys.astype(float)
    sells = sells.astype(float)

    return (buys, sells) + return_tuple


def get_buys_and_sells_stoch(X, aggregate_N, w, th, as_boolean=False, from_minute=True):
    w = aggregate_N * w
    if from_minute:
        w *= 60

    # start from index 1 so that N is the same as in the previous get_buys_and_sells function
    stoch = stochastic_oscillator(X[1:, :], w)
    N = stoch.shape[0]

    diff = np.diff(X[:, 0])
    diff = diff[-N:]

    buys = (stoch <= th) & (diff > 0)
    sells = (stoch >= 1 - th) & (diff < 0)

    return_tuple = (N,)

    if as_boolean:
        if N == 1:
            return (buys[0], sells[0]) + return_tuple
        else:
            return (buys, sells) + return_tuple

    buys = buys.astype(float)
    sells = sells.astype(float)

    return (buys, sells) + return_tuple
