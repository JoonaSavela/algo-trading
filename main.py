import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
import numpy as np
import pandas as pd
import time
from utils import *
from model import *
from optimize import *
from tqdm import tqdm
from parameter_search import *
from parameters import commissions, spread, spread_bear, spread_bull
from itertools import product

# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: find take profit parameters using highs instead of closes

# TODO: try stop loss orders (must use lows)

# TODO: make take profit parameters dependent on long term trend (e.g. 5-10x w)

# TODO: make algorithm long-biased (but tries short side frequently)

def get_trade_lengths(entries, exits, N):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    res = []

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        res.append(exit_i - entry_i)

    return res

def get_extreme_prices_within_ma(X_orig_agg, entries, exits, N, w, long = True):
    entries_idx, _ = get_entry_and_exit_idx(entries, exits, N)

    res = []

    for entry_i in entries_idx:
        idx = np.arange(entry_i, entry_i + w)
        if long:
            extreme_i = np.argmax(X_orig_agg[idx, 0])
        else:
            extreme_i = np.argmin(X_orig_agg[idx, 0])
        extreme = (X_orig_agg[idx[extreme_i], 0] / X_orig_agg[idx[-1], 0]) ** (1 / (extreme_i + 1))
        res.append(extreme)

    return res

def asdf():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)
    X_orig = X

    aggregate_N, w, m, m_bear = (3, 16, 3, 3)
    N_repeat = 100
    randomize = True

    long_trade_wealths = []
    max_long_trade_wealths = []
    long_trade_lengths = []
    long_trade_extremes = []

    short_trade_wealths = []
    max_short_trade_wealths = []
    short_trade_lengths = []
    short_trade_extremes = []

    X_bear = get_multiplied_X(X, -m_bear)
    if m > 1:
        X = get_multiplied_X(X, m)

    for n in tqdm(range(N_repeat), disable = N_repeat == 1):
        rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
        if rand_N > 0:
            X1 = X[:-rand_N, :]
            X1_orig = X_orig[:-rand_N, :]
            X1_bear = X_bear[:-rand_N, :]
        else:
            X1 = X
            X1_orig = X_orig
            X1_bear = X_bear
        X1_agg = aggregate(X1, aggregate_N)
        X1_orig_agg = aggregate(X1_orig, aggregate_N)
        X1_bear_agg = aggregate(X1_bear, aggregate_N)

        buys, sells, N = get_buys_and_sells(X1_orig_agg, w, True)

        X1_agg = X1_agg[-N:, :]
        X1_bear_agg = X1_bear_agg[-N:, :]

        wealths = get_wealths(
            X1_agg,
            buys,
            sells,
            commissions = commissions,
            spread_bull = spread if m <= 1.0 else spread_bull,
            X_bear = X1_bear_agg,
            spread_bear = spread_bear
        )

        append_trade_wealths(wealths, buys, sells, N, long_trade_wealths, max_long_trade_wealths)
        append_trade_wealths(wealths, sells, buys, N, short_trade_wealths, max_short_trade_wealths)

        long_trade_lengths1 = get_trade_lengths(buys, sells, N)
        long_trade_lengths.extend(long_trade_lengths1)
        short_trade_lengths1 = get_trade_lengths(sells, buys, N)
        short_trade_lengths.extend(short_trade_lengths1)

        long_extremes = get_extreme_prices_within_ma(X1_orig_agg, buys, sells, N, w, long = True)
        long_trade_extremes.extend(long_extremes)
        short_extremes = get_extreme_prices_within_ma(X1_orig_agg, sells, buys, N, w, long = False)
        short_trade_extremes.extend(short_extremes)

    print(np.corrcoef(long_trade_extremes, long_trade_wealths))
    print(np.corrcoef(short_trade_extremes, short_trade_wealths))

    plt.plot(long_trade_extremes, long_trade_wealths, 'g.', markersize = 3)
    # plt.plot(long_trade_extremes, max_long_trade_wealths, 'g.')
    # plt.plot(long_trade_extremes, long_trade_lengths, 'g.')
    # plt.plot(long_trade_lengths, long_trade_wealths, 'g.')
    # plt.plot(long_trade_lengths, max_long_trade_wealths, 'c.')
    # plt.show()

    plt.plot(short_trade_extremes, short_trade_wealths, 'r.', markersize = 3)
    # plt.plot(short_trade_extremes, max_short_trade_wealths, 'r.')
    # plt.plot(short_trade_extremes, short_trade_lengths, 'r.')
    # plt.plot(short_trade_lengths, short_trade_wealths, 'r.')
    # plt.plot(short_trade_lengths, max_short_trade_wealths, 'm.')
    plt.show()


def wave_visualizer(params, short = True, take_profit = True):
    aggregate_N, w, m, m_bear, take_profit_long, take_profit_short = params

    x = np.arange(1, 20 * w)
    A = 1
    omega = 0.5

    X = (x / w + A * np.sin(omega * np.pi * x / w)) / 100 + 1
    X = X.reshape(-1, 1)
    X_orig = X

    if short:
        X_bear = get_multiplied_X(X, -m_bear)
    if m > 1:
        X = get_multiplied_X(X, m)

    buys, sells, N = get_buys_and_sells(X, w)
    X = X[-N:, :]
    X_orig = X_orig[-N:, :]
    X_bear = X_bear[-N:, :]

    wealths = get_wealths(
        X,
        buys,
        sells,
        commissions = commissions,
        spread_bull = spread if m <= 1.0 else spread_bull,
        X_bear = X_bear,
        spread_bear = spread_bear
    )

    if take_profit:
        transform_wealths(wealths, buys, sells, N, take_profit_long, commissions, spread if m <= 1.0 else spread_bull)
        transform_wealths(wealths, sells, buys, N, take_profit_short, commissions, spread_bear)

    plt.plot(X_orig / X_orig[0, 0])
    plt.plot(wealths)
    if np.log10(wealths[-1]) >= 1:
        plt.yscale('log')
    plt.show()


def main():
    # asdf()
    # wave_visualizer((3, 16, 3, 3, 1.69, 2.13))

    # take_profit_long, take_profit_short = \
    # get_take_profits([
    #                     # (4, 12, 1, 3),
    #                     (3, 16, 1, 3),
    #                     (3, 16, 3, 3), # best
    #                   ],
    #                   short = True,
    #                   N_repeat = 500,
    #                   randomize = True,
    #                   step = 0.01,
    #                   verbose = True)

    plot_performance([
                        (3, 16, 1, 3, 1.2, 2.14),
                        (3, 16, 3, 3, 1.69, 2.13), # best
                        # (4, 12, 1, 3, 1.19, 2.14),
                      ],
                      N_repeat = 1,
                      short = True,
                      take_profit = True)

    # plot_displacement([
    #                     (3, 16, 1, 3, 1.2, 2.14), # best
    #                     (3, 16, 3, 3, 1.69, 2.13),
    #                   ])





if __name__ == "__main__":
    main()
