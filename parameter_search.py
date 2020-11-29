from strategy import *
import numpy as np
import pandas as pd
import glob
from utils import *
from data import *
import time
import os
from parameters import parameters, minutes_in_a_year
from tqdm import tqdm
from optimize import *
from itertools import product
from parameters import commissions, spread, spread_bear, spread_bull
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, black_litterman, \
    objective_functions
from pypfopt import BlackLittermanModel
import concurrent.futures
from bayes_opt import BayesianOptimization


def get_trade_wealths_dicts(
        rand_N,
        X,
        X_bear,
        short,
        aggregate_N,
        buys_orig,
        sells_orig,
        N_orig,
        sides,
        wealth_categories):

    trade_wealths_dict = {}
    sub_trade_wealths_dict = {}

    for side in sides:
        trade_wealths_dict[side] = {}
        sub_trade_wealths_dict[side] = {}

        for category in wealth_categories:
            trade_wealths_dict[side][category] = []
            sub_trade_wealths_dict[side][category] = []

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
    total_months = n_months

    buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)
    N_transactions_buy = len(buys_idx) * 2
    N_transactions_sell = len(sells_idx) * 2

    side_tuples = [
        ('long', X1_agg, buys, sells),
    ]

    if short:
        side_tuples.append(('short', X1_bear_agg, sells, buys))

    for side, X1, entries, exits in side_tuples:
        for category, category_trade_wealths in zip(wealth_categories, get_trade_wealths(X1, entries, exits, N)):
            trade_wealths_dict[side][category].extend(category_trade_wealths)

        for category, category_sub_trade_wealths in zip(wealth_categories, get_sub_trade_wealths(X1, entries, exits, N)):
            sub_trade_wealths_dict[side][category].extend(category_sub_trade_wealths)


    return trade_wealths_dict, sub_trade_wealths_dict, N_transactions_buy, N_transactions_sell, total_months


def get_stop_loss_take_profit_wealth(
    side,
    stop_loss_take_profit_type,
    trade_wealths_dict,
    sub_trade_wealths_dict,
    take_profit_candidates,
    stop_loss_candidates
):
    if stop_loss_take_profit_type == 'take_profit':
        return np.array(list(map(lambda x:
            get_take_profit_wealths_from_trades(
                trade_wealths_dict[side]['base'],
                trade_wealths_dict[side]['max'],
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_total = True,
                return_as_log = True
            ),
            take_profit_candidates
        )))

    elif stop_loss_take_profit_type == 'stop_loss':
        return np.array(list(map(lambda x:
            get_stop_loss_wealths_from_trades(
                trade_wealths_dict[side]['base'],
                trade_wealths_dict[side]['min'],
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_total = True,
                return_as_log = True
            ),
            stop_loss_candidates
        )))

    elif stop_loss_take_profit_type == 'trailing':
        return np.array(list(map(lambda x:
            get_trailing_stop_loss_wealths_from_sub_trades(
                sub_trade_wealths_dict[side]['base'],
                sub_trade_wealths_dict[side]['min'],
                sub_trade_wealths_dict[side]['max'],
                x,
                total_months = 1,
                commissions = 0.0,
                spread = 0.0,
                return_total = True,
                return_as_log = True
            ),
            stop_loss_candidates
        )))

    # TODO: raise a warning
    return np.array([1.0])


def skip_condition(strategy_type, frequency, args):
    aggregate_N, w = args[:2]
    test_value = aggregate_N * w * 60 / 10000 - 1
    if frequency == 'high' and test_value >= 0:
        return True, -test_value
    elif frequency == 'low' and test_value < 0:
        return True, test_value

    if strategy_type == 'macross':
        aggregate_N, w, w2 = args

        test_value = w / w2 - 1
        if test_value <= 0:
            return True, test_value

    return False, 0


def choose_get_buys_and_sells_fn(strategy_type):
    if strategy_type == 'ma':
        return get_buys_and_sells_ma
    elif strategy_type == 'stoch':
        return get_buys_and_sells_stoch
    elif strategy_type == 'macross':
        return get_buys_and_sells_macross
    else:
        raise ValueError("strategy_type")


def get_semi_adaptive_wealths(
    total_balance,
    potential_balances,
    potential_spreads,
    best_stop_loss_take_profit_wealths_dict,
    N_transactions_buy,
    N_transactions_sell,
    total_months,
    debug = False

):
    stop_loss_take_profit_wealth = 1

    for i in range(12):
        # TODO: raise warning if spread > max(potential_spreads)
        spread = potential_spreads[np.argmin(np.abs(potential_balances - total_balance * stop_loss_take_profit_wealth))]
        if debug:
            print("Spread:", spread)

        for side, v in best_stop_loss_take_profit_wealths_dict.items():
            k, param, wealth = v
            N_transactions = N_transactions_buy if side == 'long' else N_transactions_sell

            stop_loss_take_profit_wealth *= apply_commissions_and_spreads_fast(
                np.log(wealth),
                N_transactions / total_months,
                commissions,
                spread,
                n_months = 1,
                from_log = True
            )

    return stop_loss_take_profit_wealth

def get_objective_function(
        args,
        strategy_type,
        frequency,
        N_repeat_inp,
        sides,
        X_origs,
        Xs,
        X_bears,
        short,
        step,
        stop_loss_take_profit_types,
        total_balance,
        potential_balances,
        potential_spreads,
        workers = None,
        debug = False
):
    if frequency not in ['low', 'high']:
        raise ValueError("'frequency' must be etiher 'low' or 'high'")

    get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

    if not (set(stop_loss_take_profit_types) & {'stop_loss', 'take_profit', 'trailing'}):
        raise ValueError("'stop_loss_take_profit_types' must contain 'stop_loss', 'take_profit', or 'trailing'")

    stop_loss_take_profit_types = sorted(stop_loss_take_profit_types, key = lambda x: int(x != 'trailing'))

    params_dict = {}

    aggregate_N, w = args[:2]

    skip_cond, conflict_value = skip_condition(strategy_type, frequency, args)
    if skip_cond:
        return {'objective': conflict_value}

    if N_repeat_inp is None:
        N_repeat = aggregate_N * 60
    else:
        N_repeat = N_repeat_inp

    N_repeat = min(N_repeat, aggregate_N * 60)
    total_months = 0

    N_transactions_buy = 0
    N_transactions_sell = 0

    trade_wealths_dict = {}
    sub_trade_wealths_dict = {}

    wealth_categories = ['base', 'min', 'max'] # TODO: better name?

    for side in sides:
        trade_wealths_dict[side] = {}
        sub_trade_wealths_dict[side] = {}

        for category in wealth_categories:
            trade_wealths_dict[side][category] = []
            sub_trade_wealths_dict[side][category] = []

    t0 = time.time()
    trade_wealths_dicts = []
    for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
        t00 = time.time()
        buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(X_orig, *args, hourly = True)
        t1 = time.time()
        t_X = t1 - t00
        if debug:
            print("get_buys_and_sells_fn:", round_to_n(t_X, 4))

        with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as executor:
            futures = [executor.submit(
                get_trade_wealths_dicts,
                rand_N,
                X,
                X_bear,
                short,
                aggregate_N,
                buys_orig,
                sells_orig,
                N_orig,
                sides,
                wealth_categories
            ) for rand_N in np.arange(0, aggregate_N * 60, aggregate_N * 60 // N_repeat)]

            for future in concurrent.futures.as_completed(futures):
                trade_wealths_dicts.append(future.result())

    for tw_dict, sub_tw_dict, N_tr_buy, N_tr_sell, tm in trade_wealths_dicts:
        N_transactions_buy += N_tr_buy
        N_transactions_sell += N_tr_sell
        total_months += tm

        for side in sides:
            for category in wealth_categories:
                trade_wealths_dict[side][category].extend(tw_dict[side][category])
                sub_trade_wealths_dict[side][category].extend(sub_tw_dict[side][category])

    t1 = time.time()
    t_X = t1 - t0
    if debug:
        print("Loop X:", round_to_n(t_X, 4))
        print()

    max_take_profit = np.max([np.max(v['max']) for v in trade_wealths_dict.values()])
    take_profit_candidates = np.arange(1.0, max_take_profit + step, step)

    stop_loss_candidates = np.arange(0.0, 1.0, step)

    stop_loss_take_profit_wealths_dict = {}
    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
        futures = {executor.submit(
            get_stop_loss_take_profit_wealth,
            side,
            stop_loss_take_profit_type,
            trade_wealths_dict,
            sub_trade_wealths_dict,
            take_profit_candidates,
            stop_loss_candidates
        ): (side, stop_loss_take_profit_type) for stop_loss_take_profit_type, side in \
                product(stop_loss_take_profit_types, sides)}

        for future in concurrent.futures.as_completed(futures):
            side, stop_loss_take_profit_type = futures[future]

            if side not in stop_loss_take_profit_wealths_dict:
                stop_loss_take_profit_wealths_dict[side] = {}

            stop_loss_take_profit_wealths_dict[side][stop_loss_take_profit_type] = future.result()

    t1 = time.time()
    t_sltp = t1 - t0
    if debug:
        print("Loop sltp:", round_to_n(t_sltp, 4))
        print()

    best_stop_loss_take_profit_wealths_dict = {}

    for side in sides:

        best_wealth = -1
        best_param = -1
        best_sltp_type = None

        for sltp_type in stop_loss_take_profit_wealths_dict[side].keys():
            wealths = apply_commissions_and_spreads_fast(
                stop_loss_take_profit_wealths_dict[side][sltp_type],
                0,
                0.0,
                0.0,
                n_months = total_months,
                from_log = True
            )
            i = np.argmax(wealths)

            candidate_params = take_profit_candidates if sltp_type == 'take_profit' else stop_loss_candidates

            stop_loss_take_profit_param = candidate_params[i]
            stop_loss_take_profit_wealth = wealths[i]

            if stop_loss_take_profit_wealth > best_wealth:
                best_wealth = stop_loss_take_profit_wealth
                best_param = stop_loss_take_profit_param
                best_sltp_type = sltp_type

        best_stop_loss_take_profit_wealths_dict[side] = (best_sltp_type, best_param, best_wealth)

    stop_loss_take_profit_wealth = get_semi_adaptive_wealths(
        total_balance,
        potential_balances,
        potential_spreads,
        best_stop_loss_take_profit_wealths_dict,
        N_transactions_buy,
        N_transactions_sell,
        total_months,
        debug = debug
    )

    stop_loss_take_profit_tuple = ()

    for k, param, wealth in best_stop_loss_take_profit_wealths_dict.values():
        stop_loss_take_profit_tuple += (k, round_to_n(param, 4))

    objective = stop_loss_take_profit_wealth ** (1 / 12)

    params_dict = {
        "objective": objective,
        "yearly profit": stop_loss_take_profit_wealth,
        "params": args + stop_loss_take_profit_tuple,
    }

    # pbar.set_description(str(round_to_n(objective_dict[middle_spread], 4)))

    return params_dict


def get_parameter_names(strategy_type):
    if strategy_type == 'ma':
        return ['aggregate_N', 'w']
    elif strategy_type == 'stoch':
        return ['aggregate_N', 'w', 'th']
    elif strategy_type == 'macross':
        return ['aggregate_N', 'w', 'w2']
    else:
        raise ValueError('strategy_type')

# TODO: possible strategy types:
#   - stoch
#   - candlestick
#   - ma crossover
#   - 2-timeframe ma
# TODO: save m and m_bear in the output somehow
# TODO: check that short = False is handled correctly
def find_optimal_aggregated_strategy(
        client,
        coin,
        bounds,
        resolutions,
        m,
        m_bear,
        frequency = 'high',
        strategy_type = 'ma',
        stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing'],
        search_method = 'gradient',
        init_args = None,
        N_iter = None,
        N_repeat_inp = None,
        objective_dict = None,
        step = 0.01,
        verbose = True,
        disable = False,
        short = True,
        Xs_index = [0, 1]
):
    parameter_names = get_parameter_names(strategy_type)

    if search_method not in ['gradient', 'BayesOpt']:
        raise ValueError('search_method')

    sides = ['long']
    if short:
        sides.append('short')

    files = glob.glob(f'data/{coin}/*.json')
    files.sort(key = get_time)

    Xs = load_all_data(files, Xs_index)

    if not isinstance(Xs, list):
        Xs = [Xs]

    N_years = max(len(X) for X in Xs) / minutes_in_a_year

    total_balance = get_total_balance(client, False)
    total_balance = max(total_balance, 1)

    potential_balances = np.logspace(np.log10(total_balance / (10 * N_years)), \
        np.log10(total_balance * 100 * N_years), int(3000 * N_years))
    potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)

    X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    X_origs = Xs
    if m > 1:
        Xs = [get_multiplied_X(X, m) for X in Xs]

    if search_method == 'gradient':
        prev_args = None
        if init_args is None:
            init_args = ()
            for param_name in parameter_names:
                search_space = np.arange(
                    bounds[param_name][0],
                    bounds[param_name][1] + resolutions[param_name],
                    resolutions[param_name]
                )
                init_args += (np.random.choice(search_space),)

            if verbose:
                print("Init args:", init_args)
                print()

        resolution_values = np.array([v for v in resolutions.values()]).reshape(1, -1)
        args = np.array(init_args)

        if objective_dict is None:
            objective_dict = {}

        while np.any(args != prev_args):
            ds = np.array(list(product([-1, 0, 1], repeat = len(bounds))))
            candidate_args = ds * resolution_values + args.reshape(1, -1)

            for i in range(len(bounds)):
                candidate_args[:, i] = np.clip(candidate_args[:, i], *bounds[parameter_names[i]])

            candidate_args = np.unique(candidate_args, axis=0)

            best_objective = -np.Inf
            best_args = None

            t0 = time.time()

            for c_args in candidate_args:
                c_args = tuple(c_args)
                if c_args not in objective_dict:
                    objective_dict[c_args] = get_objective_function(
                            c_args,
                            strategy_type,
                            frequency,
                            N_repeat_inp,
                            sides,
                            X_origs,
                            Xs,
                            X_bears,
                            short,
                            step,
                            stop_loss_take_profit_types,
                            total_balance,
                            potential_balances,
                            potential_spreads,
                            workers = None,
                            debug = False
                    )

                if objective_dict[c_args]['objective'] > best_objective:
                    best_objective = objective_dict[c_args]['objective']
                    best_args = c_args

            prev_args = np.array(args)
            args = best_args

            t1 = time.time()

            if verbose:
                print(args)
                print(objective_dict[args])
                print(round_to_n(t1 - t0, 4))
                print()

            args = np.array(args)

        args = tuple(args)

    elif search_method == 'BayesOpt':
        if verbose:
            print("Total N_iter:", N_iter + np.sqrt(N_iter).astype(int) + \
                (1 if init_args is not None else 0))
            if init_args is not None:
                print("Init args:", init_args)
            print()

        objective_dict = {}

        def get_rounded_args(args, resolutions, parameter_names):
            res = ()
            for i in range(len(args)):
                arg = args[i]

                if type(resolutions[parameter_names[i]]) == int:
                    arg = int(round(arg))

                res += (arg,)

            return res

        if strategy_type == 'ma':
            def objective_fn(aggregate_N, w):
                args = (aggregate_N, w)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                            args,
                            strategy_type,
                            frequency,
                            N_repeat_inp,
                            sides,
                            X_origs,
                            Xs,
                            X_bears,
                            short,
                            step,
                            stop_loss_take_profit_types,
                            total_balance,
                            potential_balances,
                            potential_spreads,
                            workers = None,
                            debug = False
                    )

                return objective_dict[args]['objective']
        elif strategy_type == 'stoch':
            def objective_fn(aggregate_N, w, th):
                args = (aggregate_N, w, th)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                            args,
                            strategy_type,
                            frequency,
                            N_repeat_inp,
                            sides,
                            X_origs,
                            Xs,
                            X_bears,
                            short,
                            step,
                            stop_loss_take_profit_types,
                            total_balance,
                            potential_balances,
                            potential_spreads,
                            workers = None,
                            debug = False
                    )

                return objective_dict[args]['objective']

        elif strategy_type == 'macross':
            def objective_fn(aggregate_N, w, w2):
                args = (aggregate_N, w, w2)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                            args,
                            strategy_type,
                            frequency,
                            N_repeat_inp,
                            sides,
                            X_origs,
                            Xs,
                            X_bears,
                            short,
                            step,
                            stop_loss_take_profit_types,
                            total_balance,
                            potential_balances,
                            potential_spreads,
                            workers = None,
                            debug = False
                    )

                return objective_dict[args]['objective']

        optimizer = BayesianOptimization(
            f = objective_fn,
            pbounds = bounds,
        )

        if init_args is not None:
            optimizer.probe(
                params=dict(zip(parameter_names, init_args)),
                lazy=True,
            )

        optimizer.maximize(
            init_points = np.sqrt(N_iter).astype(int),
            n_iter = N_iter,
        )

        args = tuple(v for v in optimizer.max['params'].values())
        args = get_rounded_args(args, resolutions, parameter_names)


    return objective_dict[args], objective_dict

# TODO: save order of arguments as in find_optimal_aggregated_strategy
def save_optimal_parameters(
    all_bounds,
    all_resolutions,
    coins,
    frequencies,
    strategy_types,
    stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing'],
    N_iter = None,
    m = 3,
    m_bear = 3,
    N_repeat_inp = 40,
    step = 0.01,
    skip_existing = False,
    verbose = True,
    disable = False,
    short = True,
    Xs_index = [0, 1],
    debug = False

):
    client = FtxClient(ftx_api_key, ftx_secret_key)

    if not isinstance(coins, list):
        coins = [coins]

    if not isinstance(frequencies, list):
        frequencies = [frequencies]

    if not isinstance(strategy_types, list):
        strategy_types = [strategy_types]


    for coin, frequency, strategy_type in product(coins, frequencies, strategy_types):
        if verbose:
            print(coin, frequency, strategy_type)

        options = [coin, frequency, strategy_type]

        fname = 'optim_results/' + '_'.join(options) + '.json'
        if os.path.exists(fname):
            if skip_existing:
                if verbose:
                    print('Skipping...')
                    print()
                continue

            with open(fname, 'r') as file:
                params_dict = json.load(file)
                init_args = params_dict['params'][:len(bounds)]
        else:
            init_args = None

        parameter_names = get_parameter_names(strategy_type)

        bounds = dict([(k, all_bounds[k]) for k in parameter_names])
        resolutions = dict([(k, all_resolutions[k]) for k in parameter_names])

        N_iter1 = N_iter
        if N_iter1 is None:
            N_iter1 = 100

        if len(bounds) > 2:
            N_iter1 *= 2 ** (len(bounds) - 2)

        if verbose:
            print("Bayesian Optimization...")

        params_dict_candidate, objective_dict = find_optimal_aggregated_strategy(
                client = client,
                coin = coin,
                bounds = bounds,
                resolutions = resolutions,
                m = m,
                m_bear = m_bear,
                frequency = frequency,
                strategy_type = strategy_type,
                stop_loss_take_profit_types = stop_loss_take_profit_types,
                search_method = 'BayesOpt',
                init_args = init_args,
                N_iter = N_iter1,
                N_repeat_inp = N_repeat_inp,
                objective_dict = None,
                step = step,
                verbose = verbose,
                disable = disable,
                short = short,
                Xs_index = Xs_index
        )

        if verbose:
            print("Gradient Descent...")

        params_dict_candidate, _ = find_optimal_aggregated_strategy(
                client = client,
                coin = coin,
                bounds = bounds,
                resolutions = resolutions,
                m = m,
                m_bear = m_bear,
                frequency = frequency,
                strategy_type = strategy_type,
                stop_loss_take_profit_types = stop_loss_take_profit_types,
                search_method = 'gradient',
                init_args = params_dict_candidate['params'][:len(bounds)],
                N_iter = N_iter,
                N_repeat_inp = N_repeat_inp,
                objective_dict = objective_dict,
                step = step,
                verbose = verbose,
                disable = disable,
                short = short,
                Xs_index = Xs_index
        )


        if not debug:
            are_same = True
            if init_args is not None:
                for i in range(len(params_dict_candidate['params'])):
                    if params_dict_candidate['params'][i] != params_dict['params'][i]:
                        are_same = False
                        break

            if are_same and init_args is not None:
                if verbose:
                    print("Optimal parameters did not change for:", coin, frequency, strategy_type)
            else:
                params_dict = params_dict_candidate
                with open('optim_results/' + '_'.join(options) + '.json', 'w') as file:
                    json.dump(params_dict, file, cls = NpEncoder)

        if verbose:
            print()
            print_dict(params_dict)
            print()


def get_adaptive_wealths_for_multiple_strategies(
        strategies,
        m,
        m_bear,
        weights,
        N_repeat_inp = 60,
        compress = 60,#1440,
        disable = False,
        verbose = True,
        randomize = False,
        Xs_index = [0, 1],
        debug = False
        ):
    client = FtxClient(ftx_api_key, ftx_secret_key)

    aggregate_Ns = [v['params'][0] for v in strategies.values()]
    ws = [v['params'][1] for v in strategies.values()]

    max_aggregate_N_times_w = np.max(np.prod(list(zip(aggregate_Ns, ws)), axis = 1))

    min_aggregate_N = np.min(aggregate_Ns)
    max_aggregate_N = np.max(aggregate_Ns)

    if N_repeat_inp is None:
        N_repeat = min_aggregate_N * 60
    else:
        N_repeat = N_repeat_inp

    N_repeat = min(N_repeat, min_aggregate_N * 60)

    if randomize:
        rand_Ns = np.random.choice(np.arange(0, min_aggregate_N * 60), size=N_repeat, replace=True)
    else:
        rand_Ns = np.arange(0, min_aggregate_N * 60, min_aggregate_N * 60 // N_repeat)

    Xs_dict = {}
    wealths_dict = {}

    for strategy_key, strategy in strategies.items():
        strategy_type = strategy_key.split('_')[2]
        parameter_names = get_parameter_names(strategy_type)

        args = strategy['params'][:len(parameter_names)]
        aggregate_N, w = args[:2]

        sltp_args = strategy['params'][len(parameter_names):]
        short = len(sltp_args) == 4

        get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

        coin = strategy_key.split('_')[0]
        if coin not in Xs_dict:
            files = glob.glob(f'data/{coin}/*.json')
            files.sort(key = get_time)
            Xs = load_all_data(files, Xs_index)

            if not isinstance(Xs, list):
                Xs = [Xs]

            Xs_dict[coin] = Xs

        Xs = Xs_dict[coin]

        N_years = max(len(X) for X in Xs) / minutes_in_a_year

        total_balance = get_total_balance(client, False) * weights[strategy_key]
        total_balance = max(total_balance, 1)

        potential_balances = np.logspace(np.log10(total_balance / (10 * N_years)), \
            np.log10(total_balance * 100 * N_years), int(3000 * N_years))
        potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)

        X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
        X_origs = Xs
        if m > 1:
            Xs = [get_multiplied_X(X, m) for X in Xs]

        n_months = 0

        if verbose:
            print(strategy_key, strategy['objective'], strategy['params'])

        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            min_N_orig = len(X_orig) - max_aggregate_N_times_w * 60

            buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(X_orig, *args, hourly = True)

            pbar = tqdm(rand_Ns, disable = disable)
            for rand_N in pbar:
                min_N = np.min([((min_N_orig - rand_N) // (agg_N * 60)) * agg_N * 60 - (agg_N * 60 - 1) for agg_N in aggregate_Ns])
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    if short:
                        X1_bear = X_bear

                N_skip = (N_orig - rand_N) % (aggregate_N * 60)

                start_i = N_skip + aggregate_N * 60 - 1
                idx = np.arange(start_i, N_orig, aggregate_N * 60)
                buys = buys_orig[idx]
                sells = sells_orig[idx]

                buys_candidate = np.append(np.repeat(buys[:-1], aggregate_N * 60), buys[-1])
                sells_candidate = np.append(np.repeat(sells[:-1], aggregate_N * 60), sells[-1])

                buys = buys_candidate[-min_N:]
                sells = sells_candidate[-min_N:]

                wealths = get_adaptive_wealths(
                    X = X1[-min_N:, :],
                    buys = buys,
                    sells = sells,
                    aggregate_N = aggregate_N,
                    sltp_args = sltp_args,
                    total_balance = total_balance,
                    potential_balances = potential_balances,
                    potential_spreads = potential_spreads,
                    commissions = commissions,
                    short = short,
                    X_bear = X1_bear[-min_N:, :],
                )

                if debug:
                    print(wealths[-1])
                    quick_plot(
                        xs = [X1[-min_N:, 0], wealths],
                        colors = ['k', 'g'],
                        alphas = [0.5, 0.8],
                        log = True
                    )

                idx = np.arange(0, len(wealths), compress)
                wealths = wealths[idx]
                n_months += len(wealths) * compress / (60 * 24 * 30)

                if strategy_key not in wealths_dict:
                    wealths_dict[strategy_key] = [np.log(wealths)]
                else:
                    latest_wealth = wealths_dict[strategy_key][-1][-1]
                    wealths_dict[strategy_key].append(latest_wealth + np.log(wealths))

        wealths_dict[strategy_key] = np.concatenate(wealths_dict[strategy_key])

        wealth_log = wealths_dict[strategy_key][-1]
        if verbose:
            print(np.exp(wealth_log / n_months))
            print()

        wealths_dict[strategy_key] = np.exp(wealths_dict[strategy_key])

    df = pd.DataFrame(wealths_dict)

    return df

def save_wealths(
    strategies,
    m,
    m_bear,
    weights,
    N_repeat_inp = 60,
    compress = 60,#1440,
    disable = False,
    verbose = True,
    randomize = False,
    Xs_index = [0, 1],
    debug = False
):
    df = get_adaptive_wealths_for_multiple_strategies(
        strategies = strategies,
        m = m,
        m_bear = m_bear,
        weights = weights,
        N_repeat_inp = N_repeat_inp,
        compress = compress,
        disable = disable,
        verbose = verbose,
        randomize = randomize,
        Xs_index = Xs_index,
        debug = False
    )

    # TODO: have this as a input parameter
    df.to_csv(f'optim_results/wealths_{compress}.csv')

def optimize_weights(compress, save = True, verbose = False):
    df = pd.read_csv(f'optim_results/wealths_{compress}.csv', index_col=0)

    freq = minutes_in_a_year / compress
    mu = expected_returns.mean_historical_return(df, frequency=freq)
    S = risk_models.CovarianceShrinkage(df, frequency=freq).ledoit_wolf()
    # if verbose:
    #     print(mu)
    #     print(S)

    bl = BlackLittermanModel(S, pi = np.zeros((len(df.columns),)), absolute_views = mu)
    S_post = bl.bl_cov()
    mu_post = bl.bl_returns()
    if verbose:
        print("Posterior mu and S:")
        print(mu_post)
        print(S_post)

    ef = EfficientFrontier(mu_post, S_post)
    ef.max_sharpe(0.07)
    weights_dict = ef.clean_weights()
    if verbose:
        print()
        print("Weights:")
        print(weights_dict)
        print("Performance:")

    a, b, c = ef.portfolio_performance(verbose=verbose, risk_free_rate=0.07)

    if verbose:
        weights = np.array(list(weights_dict.values()))

        X = df.values
        wealths = X[-1, :] / X[0, :]
        wealths = wealths ** (freq / len(X))
        print()
        print(wealths)
        print(np.dot(wealths, weights))

        X_diff = X[1:, :] / X[:-1, :]

        wealths = np.prod(X_diff, axis = 0) ** (freq / len(X))
        # print(wealths)

        weighted_diffs = np.matmul(X_diff, weights)
        wealth = np.prod(weighted_diffs) ** (freq / len(X))
        print(wealth)

    if save:
        with open('optim_results/weights.json', 'w') as file:
            json.dump(weights_dict, file)

    return weights_dict, a, b, c

def optimize_weights_iterative(
        coins = ['ETH', 'BTC'],
        freqs = ['low', 'high'],
        strategy_types = ['ma', 'macross'],
        weights_type = "file",
        n_iter = 20,
        compress = 60
    ):
    strategies = {}

    for coin, freq, strategy_type in product(coins, freqs, strategy_types):
        strategy_key = '_'.join([coin, freq, strategy_type])
        filename = f'optim_results/{strategy_key}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                strategies[strategy_key] = json.load(file)

    if weights_type == "file":
        filename = 'optim_results/weights.json'
        with open(filename, 'r') as file:
            weights = json.load(file)
    elif weights_type == "equal":
        weights = dict(list(zip(
            [k for k in strategies.keys()],
            np.ones((len(strategies),)) / len(strategies)
        )))
    else:
        raise ValueError('weights_type')

    weight_values = np.zeros((len(weights),))

    for n in range(n_iter):

        weight_values *= n
        weight_values += np.array(list(weights.values()))
        weight_values /= (n + 1)
        print(weight_values)
        print()
        weights = dict(list(zip(
            weights.keys(),
            weight_values
        )))

        save_wealths(
            strategies,
            m = 3,
            m_bear = 3,
            weights = weights,
            N_repeat_inp = 5,
            compress = compress,
            disable = True,
            verbose = False,
            randomize = False,
            Xs_index = [0, 1],
            debug = False
        )

        weights, a, b, c = optimize_weights(compress, save = True, verbose = False)

        print("Weights:")
        print_dict(weights)
        print("Performance:")
        print(a, b, c)
        print()

    # TODO: delete 'wealths.csv'


def get_filtered_strategies_and_weights(
    coins = ['ETH', 'BTC'],
    freqs = ['low', 'high'],
    strategy_types = ['ma', 'macross'],
    normalize = True
):
    strategies = {}

    for coin, freq, strategy_type in product(coins, freqs, strategy_types):
        strategy_key = '_'.join([coin, freq, strategy_type])
        filename = f'optim_results/{strategy_key}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                strategies[strategy_key] = json.load(file)

    if os.path.exists('optim_results/weights.json'):
        with open('optim_results/weights.json', 'r') as file:
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

    if len(weights) > 0 and normalize:
        weight_values = np.array(list(weights.values()))
        weight_values /= np.sum(weight_values)
        weights = dict(list(zip(
            weights.keys(),
            weight_values
        )))

    return strategies, weights



def plot_weighted_adaptive_wealths(
    coins = ['ETH', 'BTC'],
    freqs = ['low', 'high'],
    strategy_types = ['ma', 'macross'],
    m = 3,
    m_bear = 3,
    N_repeat = 1,
    compress = 60,
    short = True,
    randomize = True,
    Xs_index = [0, 1]
):
    plt.style.use('seaborn')

    strategies, weights = get_filtered_strategies_and_weights(
        coins,
        freqs,
        strategy_types
    )
    print_dict(strategies)

    weight_values = np.array(list(weights.values()))

    # TODO: move theis loop into get_adaptive_wealths_for_multiple_strategies?
    for i in range(N_repeat):
        df = get_adaptive_wealths_for_multiple_strategies(
            strategies,
            m,
            m_bear,
            weights,
            N_repeat_inp = 1,
            compress = compress,
            disable = True,
            verbose = False,
            randomize = randomize,
            Xs_index = Xs_index
        )

        wealths = (df.iloc[-1] / df.iloc[0])  ** (minutes_in_a_year / (len(df) * compress))
        print(wealths)

        X = df.values

        weighted_wealths = np.matmul(X, weight_values)
        wealth = (weighted_wealths[-1] / weighted_wealths[0])  ** (minutes_in_a_year / (len(df) * compress))
        print(wealth)

        X_diff = X[1:, :] / X[:-1, :]

        weighted_diffs = np.matmul(X_diff, weight_values)
        weighted_wealths2 = np.cumprod(weighted_diffs)
        wealth = weighted_wealths2[-1] ** (minutes_in_a_year / (len(df) * compress))
        print(wealth)

        plt.plot(X, 'k', alpha = 0.25 / np.sqrt(N_repeat))
        plt.plot(weighted_wealths, 'b', alpha = 1.0 / np.sqrt(N_repeat))
        plt.plot(weighted_wealths2, 'g', alpha = 1.0 / np.sqrt(N_repeat))

    plt.yscale('log')
    plt.show()

# TODO: update
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


def get_displacements(
    coins = ['ETH', 'BTC'],
    strategy_types = ['ma', 'macross'],
    m = 3,
    m_bear = 3,
    sep = 2,
    Xs_index = [0, 1],
    plot = True,
    verbose = True,
    disable = False
):
    plt.style.use('seaborn')
    client = FtxClient(ftx_api_key, ftx_secret_key)

    strategies, weights = get_filtered_strategies_and_weights(coins, ['high'], strategy_types, False)

    wealth_categories = ['base', 'min', 'max'] # TODO: better name?

    flag = True

    Xs_dict = {}
    saved_times_dict = {}
    online_times_dict = {}
    wealths_dict = {}

    for strategy_key, strategy in strategies.items():
        if verbose:
            print(strategy_key, strategy['objective'], strategy['params'])

        strategy_type = strategy_key.split('_')[2]
        parameter_names = get_parameter_names(strategy_type)

        args = strategy['params'][:len(parameter_names)]
        aggregate_N = args[0]

        sltp_args = strategy['params'][len(parameter_names):]
        short = len(sltp_args) == 4

        sides = ['long']
        if short:
            sides.append('short')

        get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

        coin = strategy_key.split('_')[0]
        if coin not in Xs_dict:
            files = glob.glob(f'data/{coin}/*.json')
            files.sort(key = get_time)
            Xs, saved_times = load_all_data(files, Xs_index, True)
            _, online_time = get_recent_data(coin, 10, 'h', 1)

            if not isinstance(Xs, list):
                Xs = [Xs]

            Xs_dict[coin] = Xs
            saved_times_dict[coin] = saved_times
            online_times_dict[coin] = online_time

        Xs = Xs_dict[coin]
        saved_times = saved_times_dict[coin]
        online_time = online_times_dict[coin]

        N_years = max(len(X) for X in Xs) / minutes_in_a_year

        total_balance = get_total_balance(client, False) * weights[strategy_key]
        total_balance = max(total_balance, 1)

        potential_balances = np.logspace(np.log10(total_balance / (10 * N_years)), \
            np.log10(total_balance * 100 * N_years), int(3000 * N_years))
        potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)

        X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
        X_origs = Xs
        if m > 1:
            Xs = [get_multiplied_X(X, m) for X in Xs]

        Ns = np.array([len(X) for X in Xs]).reshape((-1, 1))
        N = np.sum(Ns)
        len_weights = Ns / N
        if verbose and flag:
            flag = False
            print(len_weights.flatten())

        wealth_lists = []

        for i in range(len(Xs)):
            X_orig, X, X_bear = X_origs[i], Xs[i], X_bears[i]
            buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(X_orig, *args, hourly = True)

            wealth_list = []
            time_diff = ((online_time - saved_times[i]) // 60) % (aggregate_N * 60)

            for rand_N in tqdm(range(aggregate_N * 60), disable = disable):
                trade_wealths_dict, sub_trade_wealths_dict, N_transactions_buy, N_transactions_sell, total_months = get_trade_wealths_dicts(
                    rand_N,
                    X,
                    X_bear,
                    short,
                    aggregate_N,
                    buys_orig,
                    sells_orig,
                    N_orig,
                    sides,
                    wealth_categories
                )

                stop_loss_take_profit_wealths_dict = {}

                for i, side in enumerate(sides):
                    sltp_type = sltp_args[i * 2]
                    sltp_param = sltp_args[i * 2 + 1]

                    wealth_log = get_stop_loss_take_profit_wealth(
                        side,
                        sltp_type,
                        trade_wealths_dict,
                        sub_trade_wealths_dict,
                        [sltp_param],
                        [sltp_param]
                    )
                    wealth_log = wealth_log[0]

                    wealth = apply_commissions_and_spreads_fast(
                        wealth_log,
                        0,
                        0.0,
                        0.0,
                        n_months = total_months,
                        from_log = True
                    )

                    stop_loss_take_profit_wealths_dict[side] = (sltp_type, sltp_param, wealth)

                wealth = get_semi_adaptive_wealths(
                    total_balance,
                    potential_balances,
                    potential_spreads,
                    stop_loss_take_profit_wealths_dict,
                    N_transactions_buy,
                    N_transactions_sell,
                    total_months,
                    debug = False
                )
                wealth = wealth ** (1 / 12)

                wealth_list.append(wealth)

            wealth_list = np.flip(np.array(wealth_list))
            wealth_list = np.roll(wealth_list, -time_diff + 1)

            new_wealth_list = np.ones((60,))

            for n in range(aggregate_N):
                new_wealth_list *= wealth_list[n * 60:(n + 1) * 60]

            wealth_list = new_wealth_list ** (1 / aggregate_N)

            wealth_lists.append(wealth_list)

        wealth_lists = np.stack(wealth_lists)
        wealth_list = np.exp(np.sum(np.log(wealth_lists) * len_weights, axis = 0))
        wealth_i = np.argmax(wealth_list)
        if verbose:
            print(wealth_i, wealth_list[wealth_i], wealth_list[wealth_i] ** 12)
            print()

        wealths_dict[strategy_key] = wealth_list

    diff = np.arange(-sep + 1, sep)

    li = np.ones((60,)).astype(bool)
    li[diff % 60] = False

    idx = np.arange(60)
    not_processed = np.ones((len(strategies),)).astype(bool)
    keys = np.array(list(strategies.keys()))
    keys_idx = np.arange(len(strategies))

    def _helper(key, li):
        _idx = idx[li]
        wealth_i = np.argmax(wealths_dict[key][li])
        return _idx[wealth_i], wealths_dict[key][li][wealth_i]

    displacements = {}

    for i in range(len(strategies)):
        max_i = np.argmax([_helper(key, li)[1] for key in keys[not_processed]])
        max_i = keys_idx[not_processed][max_i]
        not_processed[max_i] = False
        key = keys[max_i]
        displacements[key] = _helper(key, li)

        li[(diff + displacements[key][0]) % 60] = False

    if verbose:
        print_dict(displacements)

    # TODO: plot optim results
    if plot:
        for key in strategies.keys():
            plt.plot(wealths_dict[key])
        plt.legend(list(strategies.keys()))
        for key in strategies.keys():
            i, w = displacements[key]
            plt.plot([i], [w], 'k.', alpha = 0.8)
        plt.show()

    return displacements


if __name__ == '__main__':
    pass
