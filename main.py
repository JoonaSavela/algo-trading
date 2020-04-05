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
from itertools import product

# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: include running quantile in sells?

# TODO: try only shorting?

# TODO: plot different statistics of a strategy, use them to find improvements

# TODO: find different ways of optimizing parameters?

# TODO: remove trade_wealths from input, add output?
# TODO: change name (duh)
def asdf(wealths, entries, exits, N, sd, trade_wealths, max_trade_wealths, sds):
    idx = np.arange(N)

    entries_li = np.diff(np.concatenate((np.array([0]), entries))) == 1
    entries_idx = idx[entries_li]

    exits_li = np.diff(np.concatenate((np.array([0]), exits))) == 1
    exits_idx = idx[exits_li]

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_wealths = wealths[entry_i:exit_i]
        trade_wealths.append(sub_wealths[-1] / sub_wealths[0])
        max_trade_wealths.append(sub_wealths.max() / sub_wealths[0])
        sds.append(sd[entry_i])

def get_take_profit_wealths(trade_wealths, max_trade_wealths, take_profit, total_months):
    trade_wealths = np.array(trade_wealths)
    max_trade_wealths = np.array(max_trade_wealths)
    trade_wealths[max_trade_wealths > take_profit] = take_profit

    return np.exp(np.sum(np.log(trade_wealths)) / total_months)

# TODO: change name (duh)
def qwerty():
    params_list = [
        # (4, 12, 1, 3, 0),
        (3, 16, 1, 3, 0), # best
    ]

    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    commissions = 0.00075
    short = True
    trailing_stop = False
    N_repeat = 500
    randomize = True

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for params in params_list:
        aggregate_N, w, m, m_bear, p = params

        take_profit_long_list = []
        take_profit_short_list = []

        for i, X in enumerate(Xs):
            if short:
                X_bear = get_multiplied_X(X, -m_bear)
            if m > 1:
                X = get_multiplied_X(X, m)

            long_trade_wealths = []
            max_long_trade_wealths = []
            sds_long = []

            short_trade_wealths = []
            max_short_trade_wealths = []
            sds_short = []

            total_months = 0
            total_log_wealth = 0

            for n in tqdm(range(N_repeat), disable = N_repeat == 1):
                rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                # TODO: in case of m = 3, calculate these from original X_agg?
                buys, sells, N = get_buys_and_sells(X1_agg, w)
                sd = std(X1_agg, w + 1)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                if trailing_stop and p > 0:
                    wealths = get_wealths_trailing_stop(
                        X1, X1_agg, aggregate_N, m, p, buys, sells, commissions = commissions, X_bear = X1_bear, X_bear_agg = X1_bear_agg, m_bear = m_bear
                    )
                else:
                    wealths = get_wealths(
                        X1_agg, buys, sells, m, commissions = commissions, X_bear = X1_bear_agg, m_bear = m_bear
                    )

                n_months = buys.shape[0] * aggregate_N / (24 * 30)
                total_months += n_months

                total_log_wealth += np.log(wealths[-1])

                asdf(wealths, buys, sells, N, sd, long_trade_wealths, max_long_trade_wealths, sds_long)
                asdf(wealths, sells, buys, N, sd, short_trade_wealths, max_short_trade_wealths, sds_short)

            # plt.hist(long_trade_wealths, np.sqrt(len(long_trade_wealths)).astype(int), color='g', alpha = 0.5, density = True)
            # plt.hist(short_trade_wealths, np.sqrt(len(short_trade_wealths)).astype(int) * 2, color='r', alpha = 0.5, density = True)
            # plt.show()

            # # line = [1.0, max(long_trade_wealths)]
            # line = [1.0, max(short_trade_wealths)]
            # plt.plot(short_trade_wealths, max_short_trade_wealths, 'r.', alpha = 0.5)
            # plt.plot(long_trade_wealths, max_long_trade_wealths, 'g.', alpha = 0.5)
            # plt.plot(line, line, 'k')
            # plt.show()

            take_profits_long = np.arange(1.005, max(max_long_trade_wealths) + 0.01, 0.005)
            take_profit_wealths_long = np.array(list(map(lambda x: get_take_profit_wealths(long_trade_wealths, max_long_trade_wealths, x, total_months), take_profits_long)))

            take_profits_short = np.arange(1.005, max(max_short_trade_wealths) + 0.01, 0.005)
            take_profit_wealths_short = np.array(list(map(lambda x: get_take_profit_wealths(short_trade_wealths, max_short_trade_wealths, x, total_months), take_profits_short)))

            take_profit_long = take_profits_long[np.argmax(take_profit_wealths_long)]
            take_profit_short = take_profits_short[np.argmax(take_profit_wealths_short)]

            # take_profit_long, take_profit_short = 1.1474999999999969, 2.9424999999999586
            print(take_profit_long, take_profit_short)
            take_profit_long_list.append(take_profit_long)
            take_profit_short_list.append(take_profit_short)

            take_profit_wealth = get_take_profit_wealths(long_trade_wealths, max_long_trade_wealths, take_profit_long, total_months) * \
                get_take_profit_wealths(short_trade_wealths, max_short_trade_wealths, take_profit_short, total_months)
            print(take_profit_wealth, take_profit_wealth ** 12)
            wealth = np.exp(total_log_wealth / total_months)
            print(wealth, wealth ** 12)
            print()

            plt.plot(take_profits_long, take_profit_wealths_long ** 12, 'g')
            plt.plot(take_profits_short, take_profit_wealths_short ** 12, 'r')
            plt.show()

            # # plt.plot(sds_long, long_trade_wealths, 'g.', alpha = 0.5)
            # plt.plot(sds_long, max_long_trade_wealths, 'g.', alpha = 0.5)
            # plt.show()
            # # plt.plot(sds_long, short_trade_wealths, 'r.', alpha = 0.5)
            # plt.plot(sds_long, max_short_trade_wealths, 'r.', alpha = 0.5)
            # plt.show()

        take_profit_long = sum(take_profit_long_list) / len(take_profit_long_list)
        take_profit_short = sum(take_profit_short_list) / len(take_profit_short_list)
        print(take_profit_long, take_profit_short)

def main():
    qwerty()

    # plot_displacement([
    #                     (4, 12, 1, 3, 0),
    #                     (3, 16, 1, 3, 0), # best
    #                   ])

    # plot_performance([
    #                     # (4, 12, 1, 3, 0),
    #                     (3, 16, 1, 3, 0), # best
    #                   ],
    #                   N_repeat = 500,
    #                   short = True,
    #                   trailing_stop = False)





if __name__ == "__main__":
    main()
