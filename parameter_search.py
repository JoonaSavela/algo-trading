from strategy import *
import numpy as np
import pandas as pd
import glob
from utils import *
from data import *
import os
from parameters import parameters
from tqdm import tqdm
from optimize import *
from itertools import product
from parameters import commissions, spread, spread_bear, spread_bull


# TODO: make as general as possible
def find_optimal_aggregated_strategy(
        aggregate_N_list,
        w_list,
        m_list,
        m_bear_list,
        N_repeat = 1,
        verbose = True,
        disable = False,
        randomize = False,
        short = False):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X_orig = load_all_data(test_files, 1)

    best_reward = -np.Inf
    best_wealth = 0
    best_dropdown = 0
    best_params = None

    if not short:
        m_bear_list = [1]

    aggregate_N_prev, w_prev, m_prev, m_bear_prev = -1, -1, -1, -1
    X_bear_agg = None

    X_m_dict = {}
    def create_multiplied_X(m, X):
        return get_multiplied_X(X, m)

    for params in tqdm(list(product(aggregate_N_list, w_list, m_list, m_bear_list)), disable = disable):
        aggregate_N, w, m, m_bear = params
        # TODO: make these function parameters?
        if aggregate_N * w > 80 or aggregate_N * w < 7:
            continue
        if short:
            X_bear = get_or_create(X_m_dict, -m_bear, create_multiplied_X, X_orig)
        X = get_or_create(X_m_dict, m, create_multiplied_X, X_orig)

        total_months = 0
        total_wealth_log = 0.0
        total_dropdown_log = 0.0

        for n in range(N_repeat):
            rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
            if rand_N > 0:
                X1 = X[:-rand_N, :]
                X1_orig = X_orig[:-rand_N, :]
                if short:
                    X1_bear = X_bear[:-rand_N, :]
            else:
                X1 = X
                X1_orig = X_orig
                if short:
                    X1_bear = X_bear
            X_agg = aggregate(X1, aggregate_N)
            X1_orig_agg = aggregate(X1_orig, aggregate_N)
            if short:
                X_bear_agg = aggregate(X1_bear, aggregate_N)

            buys, sells, N = get_buys_and_sells(X1_orig_agg, w)
            if N == 0:
                continue

            X_agg = X_agg[-N:, :]
            if short:
                X_bear_agg = X_bear_agg[-N:, :]
            X1 = X1[-aggregate_N * 60 * N:, :]
            if short:
                X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

            wealths = get_wealths(
                X_agg,
                buys,
                sells,
                commissions = commissions,
                spread_bull = spread if m <= 1.0 else spread_bull,
                X_bear = X_bear_agg,
                spread_bear = spread_bear
            )

            n_months = buys.shape[0] * aggregate_N / (24 * 30)
            dropdown = get_max_dropdown(wealths)

            total_wealth_log += np.log(wealths[-1])
            total_months += n_months
            total_dropdown_log += np.log(dropdown)

        wealth = np.exp(total_wealth_log / total_months)
        dropdown = np.exp(total_dropdown_log / total_months)

        if wealth * dropdown > best_reward:
            best_reward = wealth * dropdown
            best_wealth = wealth
            best_dropdown = dropdown
            best_params = params

        aggregate_N_prev, w_prev, m_prev, m_bear_prev = aggregate_N, w, m, m_bear

    if verbose:
        print(best_params)
        print(best_wealth, best_wealth ** 12)
        print(best_dropdown, best_reward)
        print()

    return best_params



def get_take_profits(params_list, short, N_repeat, randomize, step, verbose = True):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    take_profit_long_list = []
    take_profit_short_list = []

    for params in params_list:
        aggregate_N, w, m, m_bear = params

        long_trade_wealths = []
        max_long_trade_wealths = []

        short_trade_wealths = []
        max_short_trade_wealths = []

        total_months = 0
        total_log_wealth = 0

        for i, X in enumerate(Xs):
            X_orig = X
            if short:
                X_bear = get_multiplied_X(X, -m_bear)
            if m > 1:
                X = get_multiplied_X(X, m)

            for n in tqdm(range(N_repeat), disable = N_repeat == 1):
                rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    X1_orig = X_orig[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    X1_orig = X_orig
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                X1_orig_agg = aggregate(X1_orig, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells(X1_orig_agg, w)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )

                n_months = buys.shape[0] * aggregate_N / (24 * 30)
                total_months += n_months

                total_log_wealth += np.log(wealths[-1])

                append_trade_wealths(wealths, buys, sells, N, long_trade_wealths, max_long_trade_wealths)
                append_trade_wealths(wealths, sells, buys, N, short_trade_wealths, max_short_trade_wealths)

        take_profits_long = np.arange(1.0 + step, max(max_long_trade_wealths) + step*2, step)
        take_profit_wealths_long = np.array(list(map(lambda x: get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, x, total_months, commissions, spread if m <= 1.0 else spread_bull), take_profits_long)))

        take_profits_short = np.arange(1.0 + step, max(max_short_trade_wealths) + step*2, step)
        take_profit_wealths_short = np.array(list(map(lambda x: get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, x, total_months, commissions, spread_bear), take_profits_short)))

        take_profit_long = take_profits_long[np.argmax(take_profit_wealths_long)]
        take_profit_short = take_profits_short[np.argmax(take_profit_wealths_short)]

        if verbose:
            print(take_profit_long, take_profit_short)

            take_profit_wealth = get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, take_profit_long, total_months, commissions, spread if m <= 1.0 else spread_bull) * \
                get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, take_profit_short, total_months, commissions, spread_bear)
            print(take_profit_wealth, take_profit_wealth ** 12)
            wealth = np.exp(total_log_wealth / total_months)
            print(wealth, wealth ** 12)
            print()

            plt.plot(take_profits_long, take_profit_wealths_long ** 12, 'g')
            plt.plot(take_profits_short, take_profit_wealths_short ** 12, 'r')
            plt.show()

        take_profit_long_list.append(take_profit_long)
        take_profit_short_list.append(take_profit_short)

    if len(params_list) == 1:
        return take_profit_long, take_profit_short

    return take_profit_long_list, take_profit_short_list

def get_stop_loss(params_list, short, N_repeat, randomize, step, verbose = True):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    stop_loss_long_list = []
    stop_loss_short_list = []

    for params in params_list:
        aggregate_N, w, m, m_bear = params

        long_sub_trade_wealths = []
        min_long_sub_trade_wealths = []

        short_sub_trade_wealths = []
        min_short_sub_trade_wealths = []

        total_months = 0
        total_log_wealth = 0

        for i, X in enumerate(Xs):
            X_orig = X
            if short:
                X_bear = get_multiplied_X(X, -m_bear)
            if m > 1:
                X = get_multiplied_X(X, m)

            for n in tqdm(range(N_repeat), disable = N_repeat == 1):
                rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    X1_orig = X_orig[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    X1_orig = X_orig
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                X1_orig_agg = aggregate(X1_orig, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells(X1_orig_agg, w)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )

                n_months = buys.shape[0] * aggregate_N / (24 * 30)
                total_months += n_months

                total_log_wealth += np.log(wealths[-1])

                append_sub_trade_wealths(X1_agg, buys, sells, N, long_sub_trade_wealths, min_long_sub_trade_wealths)
                append_sub_trade_wealths(X1_bear_agg, sells, buys, N, short_sub_trade_wealths, min_short_sub_trade_wealths)

        stop_losses_long = np.arange(0.0, 1.0, step)
        stop_loss_wealths_long = np.array(list(map(lambda x: get_stop_loss_wealths_from_sub_trades(long_sub_trade_wealths, min_long_sub_trade_wealths, x, total_months, commissions, spread if m <= 1.0 else spread_bull), stop_losses_long)))

        stop_losses_short = np.arange(0.0, 1.0, step)
        stop_loss_wealths_short = np.array(list(map(lambda x: get_stop_loss_wealths_from_sub_trades(short_sub_trade_wealths, min_short_sub_trade_wealths, x, total_months, commissions, spread_bear), stop_losses_short)))

        stop_loss_long = stop_losses_long[np.argmax(stop_loss_wealths_long)]
        stop_loss_short = stop_losses_short[np.argmax(stop_loss_wealths_short)]

        # stw = np.concatenate(short_sub_trade_wealths)
        # plt.hist(stw, bins=100, alpha = 0.6, density=True)
        # minstw = np.concatenate(min_short_sub_trade_wealths)
        # stw[minstw < stop_loss_short] = stop_loss_short * (1 - commissions - spread) ** 2
        # plt.hist(stw, bins=100, alpha = 0.6, density=True)
        # plt.show()

        if verbose:
            print(stop_loss_long, stop_loss_short)

            stop_loss_wealth = get_stop_loss_wealths_from_sub_trades(long_sub_trade_wealths, min_long_sub_trade_wealths, stop_loss_long, total_months, commissions, spread if m <= 1.0 else spread_bull) * \
                get_stop_loss_wealths_from_sub_trades(short_sub_trade_wealths, min_short_sub_trade_wealths, stop_loss_short, total_months, commissions, spread_bear)
            print(stop_loss_wealth, stop_loss_wealth ** 12)
            wealth = np.exp(total_log_wealth / total_months)
            print(wealth, wealth ** 12)
            print()

            plt.plot(stop_losses_long, stop_loss_wealths_long ** 12, 'g')
            plt.plot(stop_losses_short, stop_loss_wealths_short ** 12, 'r')
            plt.show()

        stop_loss_long_list.append(stop_loss_long)
        stop_loss_short_list.append(stop_loss_short)

    if len(params_list) == 1:
        return stop_loss_long, stop_loss_short

    return stop_loss_long_list, stop_loss_short_list


def plot_performance(params_list, N_repeat = 1, short = False, take_profit = True, stop_loss = True, randomize = True, Xs_index = [0, 1]):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    c_list = ['g', 'c', 'm', 'r']

    Xs = load_all_data(test_files, Xs_index)

    if N_repeat > 1:
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = [6.4 * 2, 4.8])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for i, params in enumerate(params_list):
        aggregate_N, w, m, m_bear, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = params
        print(params)

        total_log_wealths = []
        total_wealths_list = []
        t_list = []
        total_log_dropdowns = []
        total_months = []

        for n in tqdm(range(N_repeat), disable = N_repeat == 1):
            prev_price = 1.0
            count = 0
            total_log_wealth = 0
            total_log_dropdown = 0
            total_months1 = 0

            for X in Xs:
                X_orig = X
                if short:
                    X_bear = get_multiplied_X(X, -m_bear)
                if m > 1:
                    X = get_multiplied_X(X, m)

                rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
                # rand_N = 20 + 60*2
                if rand_N > 0:
                    X = X[:-rand_N, :]
                    X_orig = X_orig[:-rand_N, :]
                    if short:
                        X_bear = X_bear[:-rand_N, :]
                else:
                    X = X
                    X_orig = X_orig
                    if short:
                        X_bear = X_bear
                X_agg = aggregate(X, aggregate_N)
                X_orig_agg = aggregate(X_orig, aggregate_N)
                if short:
                    X_bear_agg = aggregate(X_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells(X_orig_agg, w)

                X_agg = X_agg[-N:, :]
                if short:
                    X_bear_agg = X_bear_agg[-N:, :]
                X = X[-aggregate_N * 60 * N:, :]
                if short:
                    X_bear = X_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths(
                    X_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X_bear_agg,
                    spread_bear = spread_bear
                )

                if take_profit or stop_loss:
                    transform_wealths(wealths, X_agg, buys, sells, N, take_profit_long, stop_loss_long, commissions, spread if m <= 1.0 else spread_bull)
                    transform_wealths(wealths, X_bear_agg, sells, buys, N, take_profit_short, stop_loss_short, commissions, spread_bear)

                n_months = buys.shape[0] * aggregate_N / (24 * 30)
                dropdown, I, J = get_max_dropdown(wealths, True)

                wealth = wealths[-1] ** (1 / n_months)

                if N_repeat == 1:
                    print(wealth, wealth ** 12)
                    # dropdown = dropdown ** (1 / n_months)
                    # print(dropdown, wealth * dropdown)

                t = np.arange(N) + count
                t *= aggregate_N * 60
                count += N

                if N_repeat == 1:
                    if i == 0:
                        buys_diff = np.diff(np.concatenate([np.array([0]), buys]))
                        buys_li = buys_diff == 1.0
                        sells_li = buys_diff == -1.0
                        idx = np.arange(N)
                        plt.plot(t, X_agg[:, 0] / X_agg[0, 0] * prev_price, c='k', alpha=0.5)
                        # plt.plot(t[idx[buys_li]], X_agg[idx[buys_li], 0] / X_agg[0, 0] * prev_price, 'g.', alpha=0.85, markersize=20)
                        # plt.plot(t[idx[sells_li]], X_agg[idx[sells_li], 0] / X_agg[0, 0] * prev_price, 'r.', alpha=0.85, markersize=20)

                    total_wealths = wealths * np.exp(total_log_wealth)
                    total_wealths_list.append(total_wealths)
                    t_list.append(t)
                    plt.plot(t, total_wealths, c=c_list[i % len(c_list)], alpha=0.9 / np.sqrt(N_repeat))
                    plt.plot(t[[I, J]], total_wealths[[I, J]], 'r.', alpha=0.85, markersize=15)
                    plt.yscale('log')

                total_log_wealth += np.log(wealths[-1])
                total_log_dropdown += np.log(dropdown)
                prev_price *= X[-1, 0] / X[0, 0]
                total_months1 += n_months

            total_log_wealths.append(total_log_wealth)
            total_log_dropdowns.append(total_log_dropdown)
            total_months.append(total_months1)

        total_months = np.sum(total_months)
        total_wealth = np.exp(np.sum(total_log_wealths) / total_months)
        total_dropdown = np.exp(np.sum(total_log_dropdowns) / total_months)
        print()
        print(total_wealth, total_wealth ** 12)
        print(total_dropdown, total_wealth * total_dropdown)
        if N_repeat == 1 and len(params_list) == 1:
            total_wealths = np.concatenate(total_wealths_list)
            t = np.concatenate(t_list)
            monthly_returns = total_wealths ** (1 / ((np.arange(total_wealths.shape[0]) + 1) * aggregate_N / (24 * 30)))
            plt.plot(t, monthly_returns ** 12, c = 'b', alpha=0.9 / np.sqrt(N_repeat))
            plt.axhline(y = 1, color = 'k', alpha = 0.25)
            print(round_to_n(total_months / 12, n = 3))
        print()

        if N_repeat > 1:
            ax[0].hist(
                np.array(total_log_wealths) / np.array(total_months) * 12,
                np.sqrt(N_repeat).astype(int),
                color=c_list[i % len(c_list)],
                alpha=0.9 / np.sqrt(len(params_list)),
                density = True
            )
            # ax[0].set_xscale('log')
            ax[1].hist(
                np.exp((np.array(total_log_wealths) + np.array(total_log_dropdowns)) / np.array(total_months)),
                np.sqrt(N_repeat).astype(int),
                color=c_list[i % len(c_list)],
                alpha=0.9 / np.sqrt(len(params_list)),
                density = True
            )

    plt.show()

def plot_displacement(params_list, short = True, take_profit = True, stop_loss = True, Xs_index = [0, 1]):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    Xs, time1 = load_all_data(test_files, Xs_index, True)
    _, time2 = get_recent_data('ETH', 10, 'h', 1)

    Ns = np.array(list(map(lambda x: x.shape[0], Xs))).reshape((-1, 1))
    N = np.sum(Ns)
    weights = Ns / N
    print(weights.flatten())

    if not isinstance(Xs, list):
        Xs = [Xs]
        time1 = [time1]

    for params in params_list:
        aggregate_N, w, m, m_bear, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = params

        wealth_lists = []
        for i, X in enumerate(Xs):
            X_orig = X
            if short:
                X_bear = get_multiplied_X(X, -m_bear)
            if m > 1:
                X = get_multiplied_X(X, m)

            wealth_list = []
            time_diff = ((time2 - time1[i]) // 60) % (aggregate_N * 60)
            # print(time_diff)

            for rand_N in tqdm(range(aggregate_N * 60)):
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    X1_orig = X_orig[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    X1_orig = X_orig
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                X1_orig_agg = aggregate(X1_orig, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells(X1_orig_agg, w)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )

                if take_profit or stop_loss:
                    transform_wealths(wealths, X1_agg, buys, sells, N, take_profit_long, stop_loss_long, commissions, spread if m <= 1.0 else spread_bull)
                    transform_wealths(wealths, X1_bear_agg, sells, buys, N, take_profit_short, stop_loss_short, commissions, spread_bear)

                n_months = buys.shape[0] * aggregate_N / (24 * 30)

                wealth = wealths[-1] ** (1 / n_months)

                # TODO: calculate std here?

                wealth_list.append(wealth)

            size = aggregate_N * 60
            wealth_list = np.flip(np.array(wealth_list))
            wealth_list = np.roll(wealth_list, -time_diff + 1)

            # TODO: include trade profit standard deviation in the plots?
            new_wealth_list = np.ones((60,))

            for n in range(aggregate_N):
                new_wealth_list *= wealth_list[n * 60:(n + 1) * 60]

            wealth_list = new_wealth_list ** (1 / aggregate_N)

            wealth_lists.append(wealth_list)

        wealth_lists = np.stack(wealth_lists)
        wealth_list = np.exp(np.sum(np.log(wealth_lists) * weights, axis = 0))
        wealth_i = np.argmax(wealth_list)
        print(wealth_i, wealth_list[wealth_i], wealth_list[wealth_i] ** 12)
        plt.plot(wealth_list)
    plt.show()


if __name__ == '__main__':
    aggregate_N_list = range(1, 13)
    w_list = range(1, 51)
    # m_list = range(1, 4, 2)
    m_list = [1, 3]
    m_bear_list = [3]

    short = True

    params = find_optimal_aggregated_strategy(
                aggregate_N_list,
                w_list,
                m_list,
                m_bear_list,
                N_repeat = 40,
                verbose = True,
                disable = False,
                randomize = True,
                short = short,
    )

    take_profits = get_take_profits([
                        params
                      ],
                      short = short,
                      N_repeat = 500,
                      randomize = True,
                      step = 0.01,
                      verbose = False)

    plot_performance([
                        params + take_profits,
                        # (3, 16, 1, 3, 1.2, 2.14),
                        (3, 16, 3, 3, 1.69, 2.13),
                      ],
                      N_repeat = 500,
                      short = short,
                      take_profit = True)
