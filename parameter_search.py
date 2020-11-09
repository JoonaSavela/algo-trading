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
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient


# TODO: make as general as possible
#   - instead of arg get_buys_and_sells_fn, have an argument for strategy type
#     (and then choose get_buys_and_sells_fn accordingly)
# TODO: multiple types of stop loss and take profit
#   - only take profit (current)
#   - only stop loss
#   - both
#   - only high frequency stop loss (current stop loss)
#   - only trailing stop loss
# TODO: save m and m_bear in the output somehow
# TODO: handle short = False
def find_optimal_aggregated_strategy_new(
        coin,
        aggregate_N_list,
        w_list,
        m,
        m_bear,
        objective_fn,
        spreads,
        frequency = 'high',
        strategy_type = 'ma',
        N_repeat_inp = None,
        step = 0.01,
        verbose = True,
        disable = False,
        short = True,
        Xs_index = [0, 1]):

    if strategy_type == 'ma':
        get_buys_and_sells_fn = get_buys_and_sells2
    else:
        raise ValueError("strategy_type")

    if frequency not in ['low', 'high']:
        raise ValueError("'frequency' must be etiher 'low' or 'high'")

    print(coin, frequency, strategy_type)

    files = glob.glob(f'data/{coin}/*.json')
    files.sort(key = get_time)

    Xs = load_all_data(files, Xs_index)

    if not isinstance(Xs, list):
        Xs = [Xs]

    X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    X_origs = Xs
    if m > 1:
        Xs = [get_multiplied_X(X, m) for X in Xs]

    params_dict = {}
    objective_dict = {}

    for spread in spreads:
        # wealths_dict[spread] = []
        objective_dict[spread] = -np.Inf

    middle_spread = spreads[len(spreads) // 2]
    pbar = tqdm(list(product(aggregate_N_list, w_list)), desc = str(objective_dict[middle_spread]), disable = disable)
    for aggregate_N, w in pbar:
        high_freq_skip_condition = aggregate_N * w * 60 > 10000
        if (frequency == 'high' and high_freq_skip_condition) or \
                (frequency == 'low' and not high_freq_skip_condition):
            continue

        if N_repeat_inp is None:
            N_repeat = aggregate_N * 60
        else:
            N_repeat = N_repeat_inp

        N_repeat = min(N_repeat, aggregate_N * 60)
        total_months = 0

        N_transactions_buy = 0
        N_transactions_sell = 0

        all_sub_trade_wealths_long = []
        all_min_sub_trade_wealths_long = []

        all_sub_trade_wealths_short = []
        all_min_sub_trade_wealths_short = []

        all_trade_wealths_long = []
        all_max_trade_wealths_long = []

        all_trade_wealths_short = []
        all_max_trade_wealths_short = []

        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(X_orig, aggregate_N * 60 * w)

            for rand_N in np.arange(0, aggregate_N * 60, aggregate_N * 60 // N_repeat):
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1[:, :3], aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear[:, :3], aggregate_N)

                N_skip = (N_orig - rand_N) % (aggregate_N * 60)

                start_i = N_skip + aggregate_N * 60 - 1
                idx = np.arange(start_i, N_orig, aggregate_N * 60)

                buys = buys_orig[idx]
                sells = sells_orig[idx]
                N = buys.shape[0]

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]

                n_months = N * aggregate_N / (24 * 30)
                total_months += n_months

                buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)
                N_transactions_buy += len(buys_idx) * 2
                N_transactions_sell += len(sells_idx) * 2

                sub_trade_wealths_long, min_sub_trade_wealths_long = get_sub_trade_wealths(X1_agg, buys, sells, N)
                sub_trade_wealths_short, min_sub_trade_wealths_short = get_sub_trade_wealths(X1_bear_agg, sells, buys, N)
                trade_wealths_long, max_trade_wealths_long = get_trade_wealths(X1_agg, buys, sells, N)
                trade_wealths_short, max_trade_wealths_short = get_trade_wealths(X1_bear_agg, sells, buys, N)

                all_sub_trade_wealths_long.extend(sub_trade_wealths_long)
                all_min_sub_trade_wealths_long.extend(min_sub_trade_wealths_long)

                all_sub_trade_wealths_short.extend(sub_trade_wealths_short)
                all_min_sub_trade_wealths_short.extend(min_sub_trade_wealths_short)

                all_trade_wealths_long.extend(trade_wealths_long)
                all_max_trade_wealths_long.extend(max_trade_wealths_long)

                all_trade_wealths_short.extend(trade_wealths_short)
                all_max_trade_wealths_short.extend(max_trade_wealths_short)

        stop_losses = np.arange(0.0, 1.0, step)
        stop_loss_wealths_long = np.array(list(map(lambda x:
            get_stop_loss_wealths_from_sub_trades(
                all_sub_trade_wealths_long,
                all_min_sub_trade_wealths_long,
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_as_log = True,
                return_N_transactions=True
            ),
            stop_losses
        )))
        N_stop_loss_transactions_buy = stop_loss_wealths_long[:, 1]
        stop_loss_wealths_long = stop_loss_wealths_long[:, 0]

        stop_loss_wealths_short = np.array(list(map(lambda x:
            get_stop_loss_wealths_from_sub_trades(
                all_sub_trade_wealths_short,
                all_min_sub_trade_wealths_short,
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_as_log = True,
                return_N_transactions=True
            ),
            stop_losses
        )))
        N_stop_loss_transactions_sell = stop_loss_wealths_short[:, 1]
        stop_loss_wealths_short = stop_loss_wealths_short[:, 0]

        take_profits_long = np.arange(1.0 + step, max(max_trade_wealths_long) + step*2, step)
        take_profit_wealths_long = np.array(list(map(lambda x:
            get_take_profit_wealths_from_trades(
                all_trade_wealths_long,
                all_max_trade_wealths_long,
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_as_log = True
            ),
            take_profits_long
        )))

        take_profits_short = np.arange(1.0 + step, max(max_trade_wealths_short) + step*2, step)
        take_profit_wealths_short = np.array(list(map(lambda x:
            get_take_profit_wealths_from_trades(
                all_trade_wealths_short,
                all_max_trade_wealths_short,
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_as_log = True
            ),
            take_profits_short
        )))

        for spread in spreads:
            stop_loss_wealths_long_with_spread = np.zeros_like(stop_loss_wealths_long)
            stop_loss_wealths_short_with_spread = np.zeros_like(stop_loss_wealths_short)
            for i in range(len(stop_losses)):
                stop_loss_wealths_long_with_spread[i] = apply_commissions_and_spreads_fast(
                    stop_loss_wealths_long[i],
                    N_transactions_buy + N_stop_loss_transactions_buy[i],
                    commissions,
                    spread,
                    n_months = total_months,
                    from_log = True
                )
                stop_loss_wealths_short_with_spread[i] = apply_commissions_and_spreads_fast(
                    stop_loss_wealths_short[i],
                    N_transactions_sell + N_stop_loss_transactions_sell[i],
                    commissions,
                    spread,
                    n_months = total_months,
                    from_log = True
                )

            take_profit_wealths_long_with_spread = np.zeros_like(take_profit_wealths_long)
            for i in range(len(take_profit_wealths_long)):
                take_profit_wealths_long_with_spread[i] = apply_commissions_and_spreads_fast(
                    take_profit_wealths_long[i],
                    N_transactions_buy,
                    commissions,
                    spread,
                    n_months = total_months,
                    from_log = True
                )

            take_profit_wealths_short_with_spread = np.zeros_like(take_profit_wealths_short)
            for i in range(len(take_profit_wealths_short)):
                take_profit_wealths_short_with_spread[i] = apply_commissions_and_spreads_fast(
                    take_profit_wealths_short[i],
                    N_transactions_sell,
                    commissions,
                    spread,
                    n_months = total_months,
                    from_log = True
                )

            stop_loss_long = stop_losses[np.argmax(stop_loss_wealths_long_with_spread)]
            stop_loss_short = stop_losses[np.argmax(stop_loss_wealths_short_with_spread)]

            take_profit_long = take_profits_long[np.argmax(take_profit_wealths_long_with_spread)]
            take_profit_short = take_profits_short[np.argmax(take_profit_wealths_short_with_spread)]

            if np.max(stop_loss_wealths_long_with_spread) > np.max(take_profit_wealths_long_with_spread):
                take_profit_long = np.Inf
            else:
                stop_loss_long = 0

            if np.max(stop_loss_wealths_short_with_spread) > np.max(take_profit_wealths_short_with_spread):
                take_profit_short = np.Inf
            else:
                stop_loss_short = 0

            stop_loss_take_profit_wealth = max(
                    np.max(stop_loss_wealths_long_with_spread),
                    np.max(take_profit_wealths_long_with_spread)
            ) * \
            max(
                np.max(stop_loss_wealths_short_with_spread),
                np.max(take_profit_wealths_short_with_spread)
            )

            # print(stop_loss_take_profit_wealth, stop_loss_take_profit_wealth ** 12)
            objective = objective_fn(stop_loss_take_profit_wealth)
            if objective > objective_dict[spread]:
                objective_dict[spread] = objective
                d = {
                    "objective": objective,
                    "params": (aggregate_N, w, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short),
                }
                params_dict[spread] = d

        pbar.set_description(str(round_to_n(objective_dict[middle_spread], 4)))


    return params_dict


# TODO: handle short = False
def save_wealths(
        strategies,
        m,
        m_bear,
        weights,
        N_repeat_inp = 60,
        disable = False,
        short = True,
        Xs_index = [0, 1]
        ):
    client = FtxClient(ftx_api_key, ftx_secret_key)

    aggregate_Ns = []
    ws = []

    for strategy_key, strategy in strategies.items():
        aggregate_Ns.extend([v['params'][0] for v in strategy.values()])
        ws.extend([v['params'][1] for v in strategy.values()])

    aggregate_Ns = np.unique(aggregate_Ns)
    min_aggregate_N = np.min(aggregate_Ns)
    max_aggregate_N = np.max(aggregate_Ns)

    max_aggregate_N_times_w = np.max(np.prod(list(zip(aggregate_Ns, ws)), axis = 1))

    if N_repeat_inp is None:
        N_repeat = min_aggregate_N * 60
    else:
        N_repeat = N_repeat_inp

    N_repeat = min(N_repeat, min_aggregate_N * 60)

    wealths_dict = {}

    for strategy_key, strategy in strategies.items():
        strategy_type = strategy_key.split('_')[2]

        if strategy_type == 'ma':
            get_buys_and_sells_fn = get_buys_and_sells2
        else:
            raise ValueError("strategy_type")

        coin = strategy_key.split('_')[0]
        files = glob.glob(f'data/{coin}/*.json')
        files.sort(key = get_time)

        # TODO: have some of these as input parameters
        # TODO: tie these into the length of X
        total_balance = get_total_balance(client, False) * weights[strategy_key]
        potential_balances = np.logspace(np.log10(total_balance / 100), np.log10(total_balance * 10000), 1000)
        potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)
        spread = potential_spreads[np.argmin(np.abs(potential_balances - total_balance))]

        Xs = load_all_data(files, Xs_index)

        if not isinstance(Xs, list):
            Xs = [Xs]

        X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
        X_origs = Xs
        if m > 1:
            Xs = [get_multiplied_X(X, m) for X in Xs]

        print(strategy_key)
        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            buys_orig_dict = {}
            sells_orig_dict = {}
            N_orig_dict = {}
            min_N_orig = len(X_orig) - max_aggregate_N_times_w * 60
            # print(min_N_orig)

            for spread, obj in strategy.items():
                aggregate_N, w, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = obj['params']
                buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(X_orig, aggregate_N * 60 * w)

                buys_orig_dict[spread] = buys_orig
                sells_orig_dict[spread] = sells_orig
                N_orig_dict[spread] = N_orig

            # continue
            for rand_N in tqdm(np.arange(0, min_aggregate_N * 60, min_aggregate_N * 60 // N_repeat), disable = disable):
                min_N = np.min([((min_N_orig - rand_N) // (agg_N * 60)) * agg_N * 60 for agg_N in aggregate_Ns])
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    if short:
                        X1_bear = X_bear

                buys_dict = {}
                sells_dict = {}

                for spread, obj in strategy.items():
                    aggregate_N, w, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = obj['params']

                    N_skip = (N_orig_dict[spread] - rand_N) % (aggregate_N * 60)

                    start_i = N_skip + aggregate_N * 60 - 1
                    idx = np.arange(start_i, N_orig_dict[spread], aggregate_N * 60)

                    buys_candidate = np.zeros((len(idx) * aggregate_N * 60,))
                    sells_candidate = np.zeros((len(idx) * aggregate_N * 60,))

                    idx = np.append(idx, N_orig_dict[spread] - rand_N) - start_i

                    for start, end in zip(idx[:-1], idx[1:]):
                        buys_candidate[start:end] = buys_orig_dict[spread][start + start_i]
                        sells_candidate[start:end] = sells_orig_dict[spread][start + start_i]

                    buys_dict[spread] = buys_candidate[-min_N:]
                    sells_dict[spread] = sells_candidate[-min_N:]

                wealths = get_adaptive_wealths(
                    X = X1[-min_N:, :],
                    buys_dict = buys_dict,
                    sells_dict = sells_dict,
                    strategy = strategy,
                    total_balance = total_balance,
                    potential_balances = potential_balances,
                    potential_spreads = potential_spreads,
                    commissions = commissions,
                    X_bear = X1_bear[-min_N:, :],
                )

                if rand_N > 0:
                    X1_orig = X_orig[:-rand_N, :]
                else:
                    X1_orig = X_orig
                X1_orig = X1_orig[-min_N:, :]
                plt.style.use('seaborn')
                plt.plot(X1_orig[:, 0] / X1_orig[0, 0], 'k', alpha = 0.5)
                plt.plot(wealths, 'g', alpha=0.7)
                plt.yscale('log')
                plt.show()
                # return

                if strategy_key not in wealths_dict:
                    wealths_dict[strategy_key] = [wealths]
                else:
                    latest_wealth = wealths_dict[strategy_key][-1][-1]
                    wealths_dict[strategy_key].append(latest_wealth * wealths)

        wealths_dict[strategy_key] = np.concatenate(wealths_dict[strategy_key])
        print(wealths_dict[strategy_key].shape, wealths_dict[strategy_key][-1])
        return

        print()


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

            wealths = get_wealths_fast(
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

        # if wealth * dropdown > best_reward:
        if wealth > best_reward:
            best_reward = wealth #* dropdown
            best_wealth = wealth
            best_dropdown = dropdown
            best_params = params

        # TODO: remove these as they are not used
        aggregate_N_prev, w_prev, m_prev, m_bear_prev = aggregate_N, w, m, m_bear

    if verbose:
        print(best_params)
        print(best_wealth, best_wealth ** 12)
        # print(best_dropdown, best_reward)
        print()

    return best_params



def get_take_profit(params_list, short, N_repeat, randomize, step, verbose = True):
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


def get_stop_loss_and_take_profit(params_list, short, N_repeat, randomize, step, verbose = True):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    stop_loss_long_list = []
    stop_loss_short_list = []

    take_profit_long_list = []
    take_profit_short_list = []

    X1_bear_agg = None

    for params in params_list:
        aggregate_N, w, m, m_bear = params

        long_sub_trade_wealths = []
        min_long_sub_trade_wealths = []

        short_sub_trade_wealths = []
        min_short_sub_trade_wealths = []

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
                X1_agg = aggregate(X1[:, :3], aggregate_N)
                X1_orig_agg = aggregate(X1_orig[:, :3], aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear[:, :3], aggregate_N)

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
                append_trade_wealths(wealths, buys, sells, N, long_trade_wealths, max_long_trade_wealths)
                append_trade_wealths(wealths, sells, buys, N, short_trade_wealths, max_short_trade_wealths)


        stop_losses_long = np.arange(0.0, 1.0, step)
        stop_loss_wealths_long = np.array(list(map(lambda x: get_stop_loss_wealths_from_sub_trades(long_sub_trade_wealths, min_long_sub_trade_wealths, x, total_months, commissions, spread if m <= 1.0 else spread_bull), stop_losses_long)))

        stop_losses_short = np.arange(0.0, 1.0, step)
        stop_loss_wealths_short = np.array(list(map(lambda x: get_stop_loss_wealths_from_sub_trades(short_sub_trade_wealths, min_short_sub_trade_wealths, x, total_months, commissions, spread_bear), stop_losses_short)))

        take_profits_long = np.arange(1.0 + step, max(max_long_trade_wealths) + step*2, step)
        take_profit_wealths_long = np.array(list(map(lambda x: get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, x, total_months, commissions, spread if m <= 1.0 else spread_bull), take_profits_long)))

        take_profits_short = np.arange(1.0 + step, max(max_short_trade_wealths) + step*2, step)
        take_profit_wealths_short = np.array(list(map(lambda x: get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, x, total_months, commissions, spread_bear), take_profits_short)))

        stop_loss_long = stop_losses_long[np.argmax(stop_loss_wealths_long)]
        stop_loss_short = stop_losses_short[np.argmax(stop_loss_wealths_short)]

        take_profit_long = take_profits_long[np.argmax(take_profit_wealths_long)]
        take_profit_short = take_profits_short[np.argmax(take_profit_wealths_short)]

        if np.max(stop_loss_wealths_long) > np.max(take_profit_wealths_long):
            take_profit_long = np.Inf
        else:
            stop_loss_long = 0

        if np.max(stop_loss_wealths_short) > np.max(take_profit_wealths_short):
            take_profit_short = np.Inf
        else:
            stop_loss_short = 0

        if verbose:
            print(take_profit_long, take_profit_short, stop_loss_long, stop_loss_short)

            stop_loss_take_profit_wealth = max([
                    get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, take_profit_long, total_months, commissions, spread if m <= 1.0 else spread_bull),
                    get_stop_loss_wealths_from_sub_trades(long_sub_trade_wealths, min_long_sub_trade_wealths, stop_loss_long, total_months, commissions, spread if m <= 1.0 else spread_bull)
                ]) * \
                max([
                    get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, take_profit_short, total_months, commissions, spread_bear),
                    get_stop_loss_wealths_from_sub_trades(short_sub_trade_wealths, min_short_sub_trade_wealths, stop_loss_short, total_months, commissions, spread_bear)
                ])
            print(stop_loss_take_profit_wealth, stop_loss_take_profit_wealth ** 12)
            wealth = np.exp(total_log_wealth / total_months)
            print(wealth, wealth ** 12)
            print()

            plt.plot(stop_losses_long, stop_loss_wealths_long ** 12, 'g')
            plt.plot(stop_losses_short, stop_loss_wealths_short ** 12, 'r')
            plt.show()
            plt.plot(take_profits_long, take_profit_wealths_long ** 12, 'g')
            plt.plot(take_profits_short, take_profit_wealths_short ** 12, 'r')
            plt.show()

        stop_loss_long_list.append(stop_loss_long)
        stop_loss_short_list.append(stop_loss_short)

        take_profit_long_list.append(take_profit_long)
        take_profit_short_list.append(take_profit_short)

    if len(params_list) == 1:
        return take_profit_long, take_profit_short, stop_loss_long, stop_loss_short

    return take_profit_long_list, take_profit_short_list, stop_loss_long_list, stop_loss_short_list


def plot_performance(params_list, N_repeat = 1, short = False, take_profit = True, stop_loss = True, randomize = True, Xs_index = [0, 1], N_ts_plots = 10, last_N_to_plot = None):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    c_list = ['g', 'c', 'm', 'r']

    Xs = load_all_data(test_files, Xs_index)

    if not isinstance(Xs, list):
        Xs = [Xs]

    if last_N_to_plot is not None:
        Xs = [X[-last_N_to_plot:, :] for X in Xs]

    for i, params in enumerate(params_list):
        aggregate_N, w, m, m_bear, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = params
        print(params)

        total_log_wealths = []
        total_wealths_list = []
        t_list = []
        total_log_dropdowns = []
        total_months = []

        for n in tqdm(range(N_repeat), disable = N_repeat <= N_ts_plots):
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
                X_agg = aggregate(X[:, :3], aggregate_N)
                X_orig_agg = aggregate(X_orig[:, :3], aggregate_N)
                if short:
                    X_bear_agg = aggregate(X_bear[:, :3], aggregate_N)

                buys, sells, N = get_buys_and_sells(X_orig_agg, w)

                X_agg = X_agg[-N:, :]
                X_orig_agg = X_orig_agg[-N:, :]
                if short:
                    X_bear_agg = X_bear_agg[-N:, :]
                X = X[-aggregate_N * 60 * N:, :]
                if short:
                    X_bear = X_bear[-aggregate_N * 60 * N:, :]

                # wealths = get_wealths(
                #     X_agg,
                #     buys,
                #     sells,
                #     commissions = commissions,
                #     spread_bull = spread if m <= 1.0 else spread_bull,
                #     X_bear = X_bear_agg,
                #     spread_bear = spread_bear
                # )

                wealths_fast = get_wealths_fast(
                    X_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X_bear_agg,
                    spread_bear = spread_bear
                )

                # print(wealths[-1], wealths_fast[-1])
                wealths = wealths_fast

                if take_profit or stop_loss:
                    transform_wealths(wealths, X_agg, buys, sells, N, take_profit_long, stop_loss_long, commissions, spread if m <= 1.0 else spread_bull)
                    transform_wealths(wealths, X_bear_agg, sells, buys, N, take_profit_short, stop_loss_short, commissions, spread_bear)

                n_months = buys.shape[0] * aggregate_N / (24 * 30)

                wealth = wealths[-1] ** (1 / n_months)

                # TODO: transform these into more sensible units
                t = np.arange(N) + count
                t *= aggregate_N * 60
                count += N

                if N_repeat <= N_ts_plots:
                    if i == 0 and n == 0:
                        buys_diff = np.diff(np.concatenate([np.array([0]), buys]))
                        buys_li = buys_diff == 1.0
                        sells_li = buys_diff == -1.0
                        idx = np.arange(N)
                        plt.plot(t, X_orig_agg[:, 0] / X_orig_agg[0, 0] * prev_price, c='k', alpha=0.5)
                        # plt.plot(t[idx[buys_li]], X_agg[idx[buys_li], 0] / X_agg[0, 0] * prev_price, 'g.', alpha=0.85, markersize=20)
                        # plt.plot(t[idx[sells_li]], X_agg[idx[sells_li], 0] / X_agg[0, 0] * prev_price, 'r.', alpha=0.85, markersize=20)

                    total_wealths = wealths * np.exp(total_log_wealth)
                    total_wealths_list.append(total_wealths)
                    t_list.append(t)
                    plt.plot(t, total_wealths, c=c_list[i % len(c_list)], alpha=0.9 / np.sqrt(N_repeat))
                    # plt.plot(t[[I, J]], total_wealths[[I, J]], 'r.', alpha=0.85, markersize=15)
                    plt.yscale('log')

                total_log_wealth += np.log(wealths[-1])
                prev_price *= X_orig_agg[-1, 0] / X_orig_agg[0, 0]
                total_months1 += n_months

            total_log_wealths.append(total_log_wealth)
            total_months.append(total_months1)

        total_months_array = np.array(total_months)
        total_months = np.sum(total_months)
        total_wealth = np.exp(np.sum(total_log_wealths) / total_months)
        print()
        print(total_wealth, total_wealth ** 12)
        # if N_repeat <= N_ts_plots and len(params_list) == 1:
        #     total_wealths = np.concatenate(total_wealths_list)
        #     t = np.concatenate(t_list)
        #     monthly_returns = total_wealths ** (1 / ((np.arange(total_wealths.shape[0]) + 1) * aggregate_N / (24 * 30)))
        #     plt.plot(t, monthly_returns ** 12, c = 'b', alpha=0.9 / np.sqrt(N_repeat))
        #     plt.axhline(y = 1, color = 'k', alpha = 0.25)
        #     print(round_to_n(total_months / 12, n = 3))
        print()

        if N_repeat > N_ts_plots:
            plt.hist(
                np.array(total_log_wealths) * 12 / total_months_array,
                np.sqrt(N_repeat).astype(int),
                color=c_list[i % len(c_list)],
                alpha=0.9 / np.sqrt(len(params_list)),
                density = True
            )

    plt.show()

# FIXME: output when the length of params_list is more than 1
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

    return wealth_i


if __name__ == '__main__':
    aggregate_N_list = range(1, 13)
    w_list = range(1, 51)
    m_list = [3]
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

    stop_losses_and_take_profits = get_stop_loss_and_take_profit([
                        params
                      ],
                      short = short,
                      N_repeat = 300,
                      randomize = True,
                      step = 0.005,
                      verbose = False)

    params = params + stop_losses_and_take_profits

    plot_performance([
                        params,
                        (3, 16, 3, 3, 1.79, np.Inf, 0, 0.98),
                      ],
                      N_repeat = 400,
                      short = short,
                      take_profit = True,
                      stop_loss=True)

    displacement = plot_displacement([
                        params,
                        (3, 16, 3, 3, 1.79, np.Inf, 0, 0.98),
                      ],
                      short = short)

    params = params + (displacement,)
    print(params)
