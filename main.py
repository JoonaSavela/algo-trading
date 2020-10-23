import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
import numpy as np
import pandas as pd
import time
from datetime import datetime
from utils import *
from model import *
from optimize import *
from tqdm import tqdm
from parameter_search import *
from parameters import commissions, spread, spread_bear, spread_bull
from itertools import product
import matplotlib.animation as animation



# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: train a network for predicitng the optimal take profit parameter(s)

# TODO: find take profit parameters using highs instead of closes

# TODO: make a generator for calculating wealths

# TODO: make a "daily" script that saves new optimal values periodically

# TODO: add comments; improve readability

# TODO: combine get_take_profit with get_stop_loss

def calculate_optimal_take_profit(take_profit_options, trade_wealths, max_trade_wealths):
    take_profit_options.sort()
    res = np.ones((trade_wealths.shape[0],)) * take_profit_options[0]

    for take_profit in take_profit_options[1:]:
        li = max_trade_wealths > take_profit
        res[li] = take_profit

    return res

def get_something(something, entries, exits, N):
    entries_idx, _ = get_entry_and_exit_idx(entries, exits, N)

    return something[entries_idx]

# Try to predict take_profit_short based on volatility (standard deviation)
def asdf():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    aggregate_N, w, m, m_bear = (3, 16, 3, 3)
    N_repeat = 100
    randomize = True

    long_trade_wealths = []
    max_long_trade_wealths = []

    short_trade_wealths = []
    max_short_trade_wealths = []

    short_stds = []

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for X in Xs:
        X_orig = X

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

            buys, sells, N, std = get_buys_and_sells(X1_orig_agg, w, return_std = True)

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

            short_std = get_something(std, sells, buys, N)
            short_stds.extend(short_std)

    short_trade_wealths = np.array(short_trade_wealths)
    max_short_trade_wealths = np.array(max_short_trade_wealths)
    # take_profit_options = np.array([1.69, 2.13])
    take_profit_options = max_short_trade_wealths
    # print(take_profit_options)
    optimal_take_profits = calculate_optimal_take_profit(take_profit_options, short_trade_wealths, max_short_trade_wealths)

    short_stds = np.array(short_stds)
    if take_profit_options.shape[0] <= 4:
        for take_profit in take_profit_options:
            plt.hist(
                short_stds[optimal_take_profits == take_profit],
                np.sqrt(N_repeat).astype(int) * 2,
                density = True,
                alpha = 0.9 / np.sqrt(take_profit_options.shape[0])
            )
    else:
        plt.plot(short_stds, optimal_take_profits, '.', alpha = 0.25)
    plt.show()

# Try to predict expected return based on volatility (standard deviation)
def asdf2():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    aggregate_N, w, m, m_bear, take_profit_long, take_profit_short = (3, 16, 3, 3, 1.69, 2.13)
    N_repeat = 100
    randomize = True

    long_trade_wealths = []
    max_long_trade_wealths = []

    short_trade_wealths = []
    max_short_trade_wealths = []

    short_stds = []
    long_stds = []

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for X in Xs:
        X_orig = X

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

            buys, sells, N, std = get_buys_and_sells(X1_orig_agg, w, return_std = True)

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

            short_std = get_something(std, sells, buys, N)
            short_stds.extend(short_std)

            long_std = get_something(std, buys, sells, N)
            long_stds.extend(long_std)

    long_trade_wealths = get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, take_profit_long, 1, commissions, spread if m <= 1.0 else spread_bull, return_total = False)
    short_trade_wealths = get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, take_profit_short, 1, commissions, spread_bear, return_total = False)

    long_stds = np.array(long_stds)
    short_stds = np.array(short_stds)

    plt.plot(long_stds, long_trade_wealths, 'g.', alpha = 0.1)
    plt.show()
    plt.plot(short_stds, short_trade_wealths, 'r.', alpha = 0.1)
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



def try_rounding_hours():
    seconds = time.time()
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    print(now_string)
    timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
    print(timeTo)
    print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(timeTo)))
    print()

    seconds = timeTo + 45 * 60
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    print(now_string)

    seconds += (59 - 44) * 60
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
    print(timeTo)
    print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(timeTo)))


def main():
    # 0.0 0.98 optimal
    # sll, sls = get_stop_loss(
    #     [
    #         # (3, 16, 1, 3),
    #         (3, 16, 3, 3),
    #     ],
    #     short = True,
    #     N_repeat = 100,
    #     randomize = True,
    #     step = 0.005,
    #     verbose = True)

    # get_stop_loss_and_take_profit(
    #     [
    #         # (3, 16, 1, 3),
    #         (3, 16, 3, 3),
    #     ],
    #     short = True,
    #     N_repeat = 100,
    #     randomize = True,
    #     step = 0.005,
    #     verbose = True)

    # asdf2()
    # wave_visualizer((3, 16, 3, 3, 1.69, 2.13))

    # take_profit_long, take_profit_short = \
    # get_take_profit([
    #                     # (4, 12, 1, 3),
    #                     # (4, 12, 3, 3),
    #                     # (3, 16, 1, 3),
    #                     (3, 16, 3, 3), # best
    #                   ],
    #                   short = True,
    #                   N_repeat = 200,
    #                   randomize = True,
    #                   step = 0.01,
    #                   verbose = True)

    plot_performance([
                        # (3, 16, 1, 3, 1.2, np.Inf, 0, 0.98),
                        (3, 16, 3, 3, 1.79, np.Inf, 0, 0.98), # best
                        (3, 16, 3, 3, 1.79, 2.13, 0, 0),
                      ],
                      N_repeat = 300,
                      short = True,
                      take_profit = True,
                      stop_loss = True,
                      Xs_index = [1],
                      N_ts_plots = 10,
                      last_N_to_plot = 60 * 24 * 60)

    # plot_displacement([
    #                     # (3, 16, 1, 3, 1.2, np.Inf, 0, 0.98),
    #                     (3, 16, 3, 3, 1.79, np.Inf, 0, 0.98), # best
    #                   ])





if __name__ == "__main__":
    main()
