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

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def random_search(files, n_runs, strategy_class, stop_loss_take_profit, restrictive):
    X = load_all_data(files)

    options = strategy_class.get_options(stop_loss_take_profit, restrictive)
    #print('options:', options)

    if not os.path.exists('search_results/'):
        os.mkdir('search_results/')

    filename = 'search_results/random_search_{}_{}_{}.csv'.format(
        options['name'],
        stop_loss_take_profit,
        restrictive
    )
    print(filename)

    try:
        resulting_df = pd.read_csv(filename, index_col = 0)
    except FileNotFoundError as e:
        print(e)
        resulting_df = pd.DataFrame()
    print(resulting_df)

    for run in range(n_runs):
        try:
            chosen_params = dict()

            for k, v in options.items():
                if k != 'name':
                    type, args = v
                    if type == 'range':
                        chosen_params[k] = int(np.random.choice(np.arange(*args)))
                    elif type == 'uniform':
                        chosen_params[k] = float(np.random.uniform(*args))

            strategy = strategy_class(chosen_params, stop_loss_take_profit, restrictive)

            profit, min_profit, max_profit = evaluate_strategy(X, strategy, 0, False)

            chosen_params['profit'] = float(profit)
            chosen_params['min_profit'] = float(min_profit)
            chosen_params['max_profit'] = float(max_profit)

            ser = pd.DataFrame({len(resulting_df): chosen_params}).T
            resulting_df = pd.concat([resulting_df, ser], ignore_index=True)

            print()
            print(resulting_df.loc[resulting_df.index[-1]:, ['profit', 'max_profit', 'min_profit']])
            resulting_df.to_csv(filename)
        except ValueError as e:
            print(e)

    print(resulting_df.loc[resulting_df['profit'].idxmax(), :])

def optimise_smoothing_strategy(coin, files, strategy_class, stop_loss_take_profit, restrictive, kappa, n_runs):
    X = load_all_data(files)

    if restrictive:
        starts = [0, X.shape[0] // 2]
    else:
        starts = [0]

    n_months = (X.shape[0] * len(starts) - sum(starts)) / (60 * 24 * 30)

    print('Number of months:', n_months)

    def objective_function(stop_loss,
                       decay,
                       take_profit,
                       maxlen,
                       waiting_time,
                       alpha):

        maxlen = int(maxlen)
        waiting_time = int(waiting_time)

        params = {
            'stop_loss': stop_loss,
            'decay': decay,
            'take_profit': take_profit,
            'maxlen': maxlen,
            'waiting_time': waiting_time,
            'alpha': alpha,
        }

        strategy = Smoothing_Strategy(params, stop_loss_take_profit, restrictive)
        if strategy.size() > 2000:
            return -1

        profits = []
        trades_list = []

        for start in starts:
            profit, min_profit, max_profit, trades, biggest_loss = evaluate_strategy(X, strategy, start, 0, False)

            profits.append(profit)
            trades_list.append(trades)

        profit = np.prod(profits) * biggest_loss
        trades = np.concatenate(trades_list)

        if trades.shape[0] == 0:
            return -1

        score = profit ** (1 / n_months) - 1
        if score > 0 and trades.shape[0] > 2:
            score /= np.std(trades)

        print('Profit:', profit ** (1 / n_months), 'Score:', score, 'Biggest loss:', biggest_loss)

        return score


    options = Smoothing_Strategy.get_options(stop_loss_take_profit, restrictive)

    # Bounded region of parameter space
    pbounds = {}

    for k, v in options.items():
        if k != 'name':
            type, bounds = v
            pbounds[k] = bounds # TODO: set bounds differently if not in strategy class

    # print(pbounds)

    optimizer = BayesianOptimization(
        f = objective_function,
        pbounds = pbounds,
        # random_state = 1,
    )

    filename = 'optim_results/{}_{}_{}_{}.json'.format(
        coin,
        options['name'],
        stop_loss_take_profit,
        restrictive
    )

    # load_logs(optimizer, logs=[filename])

    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    # for obj in parameters:
    #     params = obj['params']
    #     for k, v in options.items():
    #         if k != 'name' and k not in params:
    #             type, bounds = v
    #             params[k] = bounds[0]
    #
    #     keys_to_pop = []
    #     for k, v in params.items():
    #         if k not in options:
    #             keys_to_pop.append(k)
    #
    #     for k in keys_to_pop:
    #         params.pop(k)
    #
    #     optimizer.probe(
    #         params=params,
    #         lazy=True,
    #     )

    optimizer.maximize(
        init_points = int(np.sqrt(n_runs)),
        n_iter = n_runs,
    )

    # fix_optim_log(filename)

    print(optimizer.max)


def optimise(coin, files, strategy_class, stop_loss_take_profit, restrictive, kappa, n_runs, p):
    X = load_all_data(files)
    X = X[:round(p * X.shape[0]), :]

    if restrictive:
        starts = [0, X.shape[0] // 2]
    else:
        starts = [0]

    n_months = (X.shape[0] * len(starts) - sum(starts)) / (60 * 24 * 30)

    print('Number of months:', n_months)

    def objective_function(stop_loss,
                       decay,
                       take_profit,
                       maxlen,
                       waiting_time,
                       window_size1,
                       window_size2,
                       window_size,
                       look_back_size,
                       rolling_min_window_size,
                       min_threshold,
                       change_threshold,
                       buy_threshold,
                       sell_threshold,
                       ha_threshold,
                       c,
                       alpha):

        maxlen = int(maxlen)
        waiting_time = int(waiting_time)
        window_size1 = int(window_size1)
        window_size2 = int(window_size2)
        window_size = int(window_size)
        look_back_size = int(look_back_size)
        rolling_min_window_size = int(rolling_min_window_size)

        params = {
            'stop_loss': stop_loss,
            'decay': decay,
            'take_profit': take_profit,
            'maxlen': maxlen,
            'waiting_time': waiting_time,
            'window_size1': window_size1,
            'window_size2': window_size2,
            'window_size': window_size,
            'look_back_size': look_back_size,
            'rolling_min_window_size': rolling_min_window_size,
            'min_threshold': min_threshold,
            'change_threshold': change_threshold,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'ha_threshold': ha_threshold,
            'c': c,
            'alpha': alpha,
        }

        strategy = strategy_class(params, stop_loss_take_profit, restrictive)
        if strategy.size() > 2000:
            return -1

        profits = []
        trades_list = []

        for start in starts:
            profit, min_profit, max_profit, trades, biggest_loss = evaluate_strategy(X, strategy, start, 0, False)

            profits.append(profit)
            trades_list.append(trades)

        profit = np.prod(profits) * biggest_loss
        trades = np.concatenate(trades_list)

        if trades.shape[0] == 0:
            return -1

        score = profit ** (1 / n_months) - 1
        if score > 0 and trades.shape[0] > 2:
            score /= np.std(trades)

        print('Profit:', profit ** (1 / n_months), 'Score:', score, 'Biggest loss:', biggest_loss)

        return score


    options = strategy_class.get_options(stop_loss_take_profit, restrictive)

    # Bounded region of parameter space
    pbounds = {}

    for k, v in options.items():
        if k != 'name':
            type, bounds = v
            pbounds[k] = bounds # TODO: set bounds differently if not in strategy class

    # print(pbounds)

    optimizer = BayesianOptimization(
        f = objective_function,
        pbounds = pbounds,
        # random_state = 1,
    )

    filename = 'optim_results/{}_{}_{}_{}.json'.format(
        coin,
        options['name'],
        stop_loss_take_profit,
        restrictive
    )

    # load_logs(optimizer, logs=[filename])

    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    for obj in parameters:
        params = obj['params']
        for k, v in options.items():
            if k != 'name' and k not in params:
                type, bounds = v
                params[k] = bounds[0]

        keys_to_pop = []
        for k, v in params.items():
            if k not in options:
                keys_to_pop.append(k)

        for k in keys_to_pop:
            params.pop(k)

        optimizer.probe(
            params=params,
            lazy=True,
        )

    optimizer.maximize(
        init_points = int(np.sqrt(n_runs)),
        n_iter = n_runs,
    )

    # fix_optim_log(filename)

    print(optimizer.max)



def find_optimal_aggregated_strategy(aggregate_N_list, w_list, verbose = True, disable = False):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X_orig = load_all_data(test_files, 0)

    commissions = 0.00075

    best_wealth = 0
    best_w = -1
    best_aggregate_N = -1

    for aggregate_N in tqdm(aggregate_N_list, disable = disable):
        X_all = aggregate(X_orig, aggregate_N)
        for w in w_list:
            buys, sells, N = get_buys_and_sells(X_all, w)
            if N == 0:
                continue

            X = X_all[-N:, :]

            wealths, _, _, _, _ = get_wealths(
                X, buys, sells, commissions = commissions
            )

            n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

            wealth = wealths[-1] ** (1 / n_months)

            if wealth > best_wealth:
                best_wealth = wealth
                best_w = w
                best_aggregate_N = aggregate_N

    if verbose:
        print()
        # ETH: 11 4, 1.0969
        # BCH: 12 1, 1.1200
        print(best_aggregate_N // 60, best_w)
        print()

        w = best_w
        aggregate_N = best_aggregate_N

        # commissions = 0.0

        Xs = load_all_data(test_files, [0, 1])

        if not isinstance(Xs, list):
            Xs = [Xs]

        total_wealth = 1.0
        total_months = 0
        count = 0
        prev_price = 1.0

        for X in Xs:
            X_all = aggregate(X, aggregate_N)

            buys, sells, N = get_buys_and_sells(X_all, w)

            X = X_all[-N:, :]

            wealths, _, _, _, _ = get_wealths(
                X, buys, sells, commissions = commissions
            )

            n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

            wealth = wealths[-1] ** (1 / n_months)

            print(wealth, wealth ** 12)

            t = np.arange(N) + count
            count += N

            plt.plot(t, X[:, 0] / X[0, 0] * prev_price, c='k', alpha=0.5)
            plt.plot(t, (np.cumsum(ma) + 1) * prev_price, c='b', alpha=0.65)
            plt.plot(t, wealths * total_wealth, c='g')

            total_wealth *= wealths[-1]
            prev_price *= X[-1, 0] / X[0, 0]
            total_months += n_months

        total_wealth = total_wealth ** (1 / total_months)
        print()
        print(total_wealth, total_wealth ** 12)

        plt.show()

    return best_aggregate_N, best_w


def find_optimal_oco_order_params_helper(X, X_agg, w, aggregate_N, disable, return_buys_output = False):
    best_p = 0.0
    best_m = 0
    best_wealth = -1

    buys, sells, N = get_buys_and_sells(X_agg, w)

    X_agg = X_agg[-N:, :]
    X = X[-aggregate_N * N:, :]

    n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

    for m in tqdm(np.arange(1.0, 3.0, 0.1), disable = disable):
        for p in np.arange(0, 0.01, 0.00025):

            wealths, _, _, _, _ = get_wealths_oco(
                X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = False
            )

            wealth = wealths[-1] ** (1 / n_months)

            if wealth > best_wealth:
                best_wealth = wealth
                best_p = p
                best_m = m

    if return_buys_output:
        return best_m, best_p, buys, sells, N

    return best_m, best_p

def find_optimal_oco_order_params(aggregate_N, w, verbose = True, disable = False):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)

    X_agg = aggregate(X, aggregate_N)

    best_m, best_p = find_optimal_oco_order_params_helper(X, X_agg, w, aggregate_N, disable)

    if verbose:
        print()
        # 2, 0.005
        # 2.4, 0.00475
        print(best_m, best_p)
        print()

        p = best_p
        m = best_m

        # commissions = 0.0

        Xs = load_all_data(test_files, [0, 1])

        if not isinstance(Xs, list):
            Xs = [Xs]

        total_wealth = 1.0
        total_months = 0
        count = 0
        prev_price = 1.0

        for X in Xs:
            X_agg = aggregate(X, aggregate_N)

            buys, sells, N = get_buys_and_sells(X_agg, w)

            X_agg = X_agg[-N:, :]
            X = X[-aggregate_N * N:, :]

            wealths, _, _, _, _ = get_wealths_oco(
                X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = False
            )

            n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

            wealth = wealths[-1] ** (1 / n_months)

            print(wealth, wealth ** 12)

            t = np.arange(N) + count
            count += N

            plt.plot(t, X_agg[:, 0] / X_agg[0, 0] * prev_price, c='k', alpha=0.5)
            plt.plot(t, (np.cumsum(ma) + 1) * prev_price, c='b', alpha=0.65)
            plt.plot(t, wealths * total_wealth, c='g')

            total_wealth *= wealths[-1]
            prev_price *= X[-1, 0] / X[0, 0]
            total_months += n_months

        total_wealth = total_wealth ** (1 / total_months)
        print()
        print(total_wealth, total_wealth ** 12)

        plt.show()

    return best_m, best_p


def find_optimal_aggregated_oco_strategy(verbose = True, N_repeat = 1):
    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)

    commissions = 0.00075

    best_wealth = 0
    best_w = -1
    best_aggregate_N = -1
    best_p = 0.0
    best_m = 0

    for aggregate_N in tqdm(range(60, 60*13, 60)):
        _, w = find_optimal_aggregated_strategy([aggregate_N], range(1, 51), False, True)

        total_months = 0
        total_wealth = 1.0

        for n in range(N_repeat):
            rand_N = np.random.randint(aggregate_N)
            if rand_N > 0:
                X1 = X[:-rand_N, :]
            X_agg = aggregate(X1, aggregate_N)

            m, p, buys, sells, N = find_optimal_oco_order_params_helper(X1, X_agg, w, aggregate_N, True, True)

            X_agg1 = X_agg[-N:, :]
            X1 = X1[-aggregate_N * N:, :]

            wealths, _, _, _, _ = get_wealths_oco(
                X1, X_agg1, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = False
            )

            n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

            total_wealth *= wealths[-1]
            total_months += n_months

        wealth = total_wealth ** (1 / total_months)

        if wealth > best_wealth:
            best_wealth = wealth
            best_w = w
            best_aggregate_N = aggregate_N
            best_p = p
            best_m = m

    if verbose:
        print(best_aggregate_N // 60, best_w, best_m, best_p)
        print(best_wealth, best_wealth ** 12)
        print()

    return best_aggregate_N, best_w, best_m, best_p

def plot_performance(params_list, N_repeat = 1):
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    c_list = ['g', 'c', 'm', 'r']

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for i, params in enumerate(params_list):
        aggregate_N, w, m, p = params
        print(aggregate_N // 60, w, m, p)

        total_log_wealths = []
        total_months = []

        for n in tqdm(range(N_repeat), disable = N_repeat == 1):
            prev_price = 1.0
            count = 0
            total_wealth1 = 1.0
            total_months1 = 0

            for X in Xs:
                rand_N = np.random.randint(aggregate_N)
                if rand_N > 0:
                    X = X[:-rand_N, :]
                X_agg = aggregate(X, aggregate_N)

                buys, sells, N = get_buys_and_sells(X_agg, w)

                X_agg = X_agg[-N:, :]
                X = X[-aggregate_N * N:, :]

                wealths, _, _, _, _ = get_wealths_oco(
                    X, X_agg, aggregate_N, p, m, buys, sells, commissions = 0.00075, verbose = N_repeat == 1
                )

                n_months = buys.shape[0] * aggregate_N / (60 * 24 * 30)

                wealth = wealths[-1] ** (1 / n_months)

                if N_repeat == 1:
                    print(wealth, wealth ** 12)

                t = np.arange(N) + count
                t *= aggregate_N
                count += N

                if N_repeat == 1:
                    if i == 0:
                        plt.plot(t, X_agg[:, 0] / X_agg[0, 0] * prev_price, c='k', alpha=0.5)
                    # plt.plot(t, (np.cumsum(ma) + 1) * prev_price, c='b', alpha=0.65 ** i)
                    plt.plot(t, wealths * total_wealth1, c=c_list[i % len(c_list)], alpha=0.9 / np.sqrt(N_repeat))

                total_wealth1 *= wealths[-1]
                prev_price *= X[-1, 0] / X[0, 0]
                total_months1 += n_months

            total_log_wealths.append(np.log(total_wealth1))
            total_months.append(total_months1)

        total_wealth = np.exp(np.sum(total_log_wealths) / np.sum(total_months))
        print()
        print(total_wealth, total_wealth ** 12)
        print()

        if N_repeat > 1:
            plt.hist(
                np.array(total_log_wealths) / np.array(total_months),
                np.sqrt(N_repeat).astype(int),
                color=c_list[i % len(c_list)],
                alpha=0.9 / np.sqrt(len(params_list))
            )

    plt.show()

if __name__ == '__main__':
    # aggregate_N_list = range(60, 60*25, 60)
    # w_list = range(1, 51)
    #
    # aggregate_N, w = find_optimal_aggregated_strategy(aggregate_N_list, w_list, False)
    # print(aggregate_N // 60, w)
    # m, p = find_optimal_oco_order_params(aggregate_N, w, True)
    # print(m, p)

    aggregate_N, w, m, p = find_optimal_aggregated_oco_strategy(False, 40)

    plot_performance([(aggregate_N, w, m, p),
                      (5 * 60, 9, 1.0, 0.0),
                      (1 * 60, 46, 1.6, 0.0065)],
                      N_repeat = 500)

    # n_runs = 800
    # kappa = 1
    # p = 0.9
    # strategy_class = Main_Strategy
    # stop_loss_take_profit = True
    # restrictive = True
    #
    # coin = 'ETH'
    #
    # dir = 'data/' + coin + '/'
    # files = glob.glob(dir + '*.json')
    # files.sort(key = get_time)
    # # print(dir, len(files), round(len(files) * 2001 / (60 * 24)))
    #
    # optimise(coin, files, strategy_class, stop_loss_take_profit, restrictive, kappa, n_runs, p)
    # random_search(files, n_runs, strategy_class, stop_loss_take_profit, restrictive)
