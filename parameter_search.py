import numpy as np
import pandas as pd
import glob
from utils import *
from data import *
import time
import os
from tqdm import tqdm
from optimize import *
from itertools import product
from parameters import commissions, minutes_in_a_year
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, black_litterman, objective_functions
from pypfopt import BlackLittermanModel
import concurrent.futures
from bayes_opt import BayesianOptimization
import warnings
from utils import (
    choose_get_buys_and_sells_fn,
    get_filtered_strategies_and_weights,
    get_parameter_names,
)

# from scipy.optimize import basinhopping


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
    trade_wealth_categories,
):

    trade_wealths_dict = {}
    sub_trade_wealths_dict = {}

    for side in sides:
        trade_wealths_dict[side] = {}
        sub_trade_wealths_dict[side] = {}

        for category in trade_wealth_categories:
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
        ("long", X1_agg, buys, sells),
    ]

    if short:
        side_tuples.append(("short", X1_bear_agg, sells, buys))

    for side, X1, entries, exits in side_tuples:
        for category, category_trade_wealths in zip(
            trade_wealth_categories, get_trade_wealths(X1, entries, exits, N)
        ):
            trade_wealths_dict[side][category].extend(category_trade_wealths)

        for category, category_sub_trade_wealths in zip(
            trade_wealth_categories, get_sub_trade_wealths(X1, entries, exits, N)
        ):
            sub_trade_wealths_dict[side][category].extend(category_sub_trade_wealths)

    return (
        trade_wealths_dict,
        sub_trade_wealths_dict,
        N_transactions_buy,
        N_transactions_sell,
        total_months,
    )


def get_stop_loss_take_profit_wealth(
    side,
    stop_loss_take_profit_type,
    trade_wealths_dict,
    sub_trade_wealths_dict,
    take_profit_candidates,
    stop_loss_candidates,
    trail_value_recalc_period=None,
):

    if stop_loss_take_profit_type == "take_profit":
        return np.array(
            list(
                map(
                    lambda x: get_take_profit_wealths_from_trades(
                        trade_wealths_dict[side]["base"],
                        trade_wealths_dict[side]["max"],
                        x,
                        total_months=1,
                        return_total=True,
                        return_as_log=True,
                    ),
                    take_profit_candidates,
                )
            )
        )

    elif stop_loss_take_profit_type == "stop_loss":
        return np.array(
            list(
                map(
                    lambda x: get_stop_loss_wealths_from_trades(
                        trade_wealths_dict[side]["base"],
                        trade_wealths_dict[side]["min"],
                        x,
                        total_months=1,
                        return_total=True,
                        return_as_log=True,
                    ),
                    stop_loss_candidates,
                )
            )
        )

    elif stop_loss_take_profit_type == "trailing":
        return np.array(
            list(
                map(
                    lambda x: get_trailing_stop_loss_wealths_from_sub_trades(
                        sub_trade_wealths_dict[side]["base"],
                        sub_trade_wealths_dict[side]["min"],
                        sub_trade_wealths_dict[side]["max"],
                        x,
                        total_months=1,
                        trail_value_recalc_period=trail_value_recalc_period,
                        return_total=True,
                        return_as_log=True,
                    ),
                    stop_loss_candidates,
                )
            )
        )

    raise ValueError("stop_loss_take_profit_type")
    return np.array([1.0])


def skip_condition(strategy_type, frequency, args):
    aggregate_N, w = args[:2]
    test_value = (w + 1) * 60 * aggregate_N / 10000 - 1
    if frequency == "high" and test_value >= 0:
        return True, -test_value
    elif frequency == "low" and test_value < 0:
        return True, test_value

    if strategy_type == "macross":
        aggregate_N, w, w2 = args

        test_value = w / w2 - 1
        if test_value <= 0:
            return True, test_value

    return False, 0


def get_semi_adaptive_wealths(
    total_balance,
    potential_balances,
    potential_spreads,
    best_stop_loss_take_profit_wealths_dict,
    N_transactions_buy,
    N_transactions_sell,
    total_months,
    debug=False,
):

    stop_loss_take_profit_wealth = 1

    for i in range(12):
        if total_balance * stop_loss_take_profit_wealth > np.max(potential_balances):
            warnings.warn(
                "total balance times wealth is greater than the max of potential balances."
            )
            # print(i, stop_loss_take_profit_wealth, np.max(potential_balances))
        spread = potential_spreads[
            np.argmin(
                np.abs(
                    potential_balances - total_balance * stop_loss_take_profit_wealth
                )
            )
        ]
        if debug:
            print("Spread:", spread)

        for side, v in best_stop_loss_take_profit_wealths_dict.items():
            wealth = v[-1]
            if wealth > 1.0 or side == "long":
                N_transactions = (
                    N_transactions_buy if side == "long" else N_transactions_sell
                )

                stop_loss_take_profit_wealth *= apply_commissions_and_spreads(
                    np.log(wealth),
                    N_transactions / total_months,
                    commissions,
                    spread,
                    n_months=1,
                    from_log=True,
                )

    return stop_loss_take_profit_wealth


def get_objective_function(
    args,
    strategy_type,
    frequency,
    coin,
    m,
    m_bear,
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
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    workers=None,
    debug=False,
    taxes=False,
):

    sltp_results_dir = "optim_results/sltp_results/"
    if not os.path.exists(sltp_results_dir):
        os.mkdir(sltp_results_dir)

    joined_str_args = "_".join(str(arg) for arg in args)

    if frequency not in ["low", "high"]:
        raise ValueError("'frequency' must be etiher 'low' or 'high'")

    get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

    if not (
        set(stop_loss_take_profit_types) & {"stop_loss", "take_profit", "trailing"}
    ):
        raise ValueError(
            "'stop_loss_take_profit_types' must contain 'stop_loss', 'take_profit', or 'trailing'"
        )

    stop_loss_take_profit_types = sorted(
        stop_loss_take_profit_types, key=lambda x: int(x != "trailing")
    )

    params_dict = {}

    aggregate_N, w = args[:2]

    skip_cond, conflict_value = skip_condition(strategy_type, frequency, args)
    if skip_cond:
        return {"objective": conflict_value}

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

    trade_wealth_categories = ["base", "min", "max"]

    for side in sides:
        trade_wealths_dict[side] = {}
        sub_trade_wealths_dict[side] = {}

        for category in trade_wealth_categories:
            trade_wealths_dict[side][category] = []
            sub_trade_wealths_dict[side][category] = []

    t0 = time.time()
    cumulative_time = 0
    trade_wealths_dicts = []
    for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
        t00 = time.time()
        buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(
            X_orig, *args, from_minute=True
        )
        t1 = time.time()
        t_X = t1 - t00
        cumulative_time += t_X
        if debug:
            print("get_buys_and_sells_fn:", round_to_n(t_X, 4))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
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
                    trade_wealth_categories,
                )
                for rand_N in np.arange(
                    0, aggregate_N * 60, aggregate_N * 60 // N_repeat
                )
            ]

            for future in concurrent.futures.as_completed(futures):
                trade_wealths_dicts.append(future.result())

    for tw_dict, sub_tw_dict, N_tr_buy, N_tr_sell, tm in trade_wealths_dicts:
        N_transactions_buy += N_tr_buy
        N_transactions_sell += N_tr_sell
        total_months += tm

        for side in sides:
            for category in trade_wealth_categories:
                trade_wealths_dict[side][category].extend(tw_dict[side][category])
                sub_trade_wealths_dict[side][category].extend(
                    sub_tw_dict[side][category]
                )

    t1 = time.time()
    t_X = t1 - t0 - cumulative_time
    if debug:
        print("Loop trade_wealths_dicts:", round_to_n(t_X, 4))
        print()

    max_take_profit = np.max([np.max(v["max"]) for v in trade_wealths_dict.values()])

    if place_take_profit_and_stop_loss_simultaneously:
        # GOAL: dict 'best_stop_loss_take_profit_wealths_dict' with
        # best_stop_loss_take_profit_wealths_dict[side] = \
        #   (best_take_profit, best_stop_loss_type, best_stop_loss, best_wealth)
        # for side in sides (e.g. ['long', 'short'])

        # inclusive on both sides
        take_profit_limits = [0, max_take_profit + step]
        stop_loss_limits = [0, 1.0 - step]

        # loop through regular stop loss and trailing stop loss
        #   - filter such that only those in stop_loss_take_profit_types are processed
        stop_loss_types = list(
            filter(
                lambda x: x in ["stop_loss", "trailing"], stop_loss_take_profit_types
            )
        )
        stop_loss_types = list(sorted(stop_loss_types))

        if not stop_loss_types:
            stop_loss_types = ["stop_loss"]
            stop_loss_limits = [0, 0]  # equivalent to having no stop loss at all

        # lower_bounds = [take_profit_limits[0], stop_loss_limits[0]]
        # upper_bounds = [take_profit_limits[1], stop_loss_limits[1]]

        best_stop_loss_take_profit_wealths_dict = {}
        t0 = time.time()

        for side in sides:
            # TODO: move into a function

            best_take_profit = -1
            best_stop_loss_type = None
            best_stop_loss = -1
            best_wealth = -1

            if debug:
                print(side)

            # for stop_loss_type in stop_loss_types:
            #     stop_loss_type_is_trailing = stop_loss_type == 'trailing'
            #
            #     print(side, stop_loss_type)

            def sltp_objective_function(take_profit, stop_loss, stop_loss_type):
                stop_loss_type = int(round(stop_loss_type))
                stop_loss_type = stop_loss_types[stop_loss_type]

                return get_stop_loss_take_profit_wealths_from_sub_trades(
                    sub_trade_wealths_dict[side]["base"],
                    sub_trade_wealths_dict[side]["min"],
                    sub_trade_wealths_dict[side]["max"],
                    stop_loss=stop_loss,
                    stop_loss_type_is_trailing=stop_loss_type == "trailing",
                    take_profit=take_profit,
                    total_months=total_months,
                    trail_value_recalc_period=trail_value_recalc_period,
                    return_total=True,
                    return_as_log=False,
                    taxes=taxes,
                )

            pbounds = {
                "take_profit": take_profit_limits,
                "stop_loss": stop_loss_limits,
                "stop_loss_type": [0, len(stop_loss_types) - 1],
            }

            optimizer = BayesianOptimization(
                f=sltp_objective_function, pbounds=pbounds, verbose=1 if debug else 0
            )

            filename = (
                sltp_results_dir
                + f'{coin}_{frequency}_{strategy_type}_{m}_{m_bear}_{joined_str_args}_{"_".join(stop_loss_types)}_{side}.json'
            )

            if os.path.exists(filename):
                with open(filename, "r") as file:
                    prev_best_sltp_args = json.load(file)

                optimizer.probe(params=prev_best_sltp_args, lazy=True)

                init_points = 50
            else:
                init_points = 100

            optimizer.maximize(
                init_points=init_points,
                n_iter=5,
            )

            take_profit = optimizer.max["params"]["take_profit"]
            stop_loss = optimizer.max["params"]["stop_loss"]
            stop_loss_type = stop_loss_types[
                int(round(optimizer.max["params"]["stop_loss_type"]))
            ]
            wealth = optimizer.max["target"]

            with open(filename, "w") as file:
                json.dump(optimizer.max["params"], file)

                # def sltp_objective_function(x):
                #     outside_of_bounds = np.linalg.norm(np.clip(x, lower_bounds, upper_bounds) - np.array(x))
                #     if outside_of_bounds > 0:
                #         return outside_of_bounds
                #
                #     return -get_stop_loss_take_profit_wealths_from_sub_trades(
                #         sub_trade_wealths_dict[side]['base'],
                #         sub_trade_wealths_dict[side]['min'],
                #         sub_trade_wealths_dict[side]['max'],
                #         stop_loss = x[1],
                #         stop_loss_type_is_trailing = stop_loss_type_is_trailing,
                #         take_profit = x[0],
                #         total_months = total_months,
                #         trail_value_recalc_period = trail_value_recalc_period,
                #         return_total = True,
                #         return_as_log = False,
                #         taxes = taxes
                #     )
                #
                # # calculate optimal wealth and parameters
                # min_result = basinhopping(
                #     sltp_objective_function,
                #     [2, min(0.9, stop_loss_limits[1])],
                #     niter = 10,
                #     # [take_profit_limits, stop_loss_limits],
                #     # disp = debug,
                #     # workers = workers
                # )
                #
                # take_profit = min_result.x[0]
                # stop_loss = min_result.x[1]
                # wealth = -min_result.fun

            if debug:
                print(take_profit, stop_loss_type, stop_loss, wealth)

            if wealth > best_wealth:
                best_take_profit = take_profit
                best_stop_loss_type = stop_loss_type
                best_stop_loss = stop_loss
                best_wealth = wealth

            best_stop_loss_take_profit_wealths_dict[side] = (
                best_take_profit,
                best_stop_loss_type,
                best_stop_loss,
                best_wealth,
            )

        t1 = time.time()
        t_sltp = t1 - t0
        if debug:
            print("Loop sltp:", round_to_n(t_sltp, 4))
            print()

    else:
        take_profit_candidates = np.arange(1.0, max_take_profit + step, step)

        stop_loss_candidates = np.arange(0.0, 1.0 - step, step)

        stop_loss_take_profit_wealths_dict = {}
        t0 = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    get_stop_loss_take_profit_wealth,
                    side,
                    stop_loss_take_profit_type,
                    trade_wealths_dict,
                    sub_trade_wealths_dict,
                    take_profit_candidates,
                    stop_loss_candidates,
                    trail_value_recalc_period=trail_value_recalc_period,
                ): (side, stop_loss_take_profit_type)
                for stop_loss_take_profit_type, side in product(
                    stop_loss_take_profit_types, sides
                )
            }

            for future in concurrent.futures.as_completed(futures):
                side, stop_loss_take_profit_type = futures[future]

                if side not in stop_loss_take_profit_wealths_dict:
                    stop_loss_take_profit_wealths_dict[side] = {}

                stop_loss_take_profit_wealths_dict[side][
                    stop_loss_take_profit_type
                ] = future.result()

        t1 = time.time()
        t_sltp = t1 - t0
        if debug:
            print("Loop sltp 1:", round_to_n(t_sltp, 4))
            print()

        best_stop_loss_take_profit_wealths_dict = {}

        for side in sides:

            best_sltp_type = None
            best_param = -1
            best_wealth = -1

            for sltp_type in stop_loss_take_profit_wealths_dict[side].keys():
                wealths = apply_commissions_and_spreads(
                    stop_loss_take_profit_wealths_dict[side][sltp_type],
                    0,
                    0.0,
                    0.0,
                    n_months=total_months,
                    from_log=True,
                )
                i = np.argmax(wealths)

                candidate_params = (
                    take_profit_candidates
                    if sltp_type == "take_profit"
                    else stop_loss_candidates
                )

                stop_loss_take_profit_param = candidate_params[i]
                stop_loss_take_profit_wealth = wealths[i]
                if debug:
                    print(
                        side,
                        sltp_type,
                        i,
                        stop_loss_take_profit_param,
                        stop_loss_take_profit_wealth,
                    )

                if stop_loss_take_profit_wealth > best_wealth:
                    best_wealth = stop_loss_take_profit_wealth
                    best_param = stop_loss_take_profit_param
                    best_sltp_type = sltp_type

            best_stop_loss_take_profit_wealths_dict[side] = (
                best_sltp_type,
                best_param,
                best_wealth,
            )

            if debug:
                print()

        t1 = time.time()
        t_sltp = t1 - t0 - t_sltp
        if debug:
            print("Loop sltp 2:", round_to_n(t_sltp, 4))
            print()

    stop_loss_take_profit_wealth = get_semi_adaptive_wealths(
        total_balance,
        potential_balances,
        potential_spreads,
        best_stop_loss_take_profit_wealths_dict,
        N_transactions_buy,
        N_transactions_sell,
        total_months,
        debug=False,  # debug
    )

    stop_loss_take_profit_tuple = ()

    for side, v in best_stop_loss_take_profit_wealths_dict.items():
        wealth = v[-1]

        # TODO: remove "side == 'long'"; requires some additional logic along the whole pipeline
        if wealth > 1.0 or side == "long":
            new_tuple = tuple(
                round_to_n(x, 4) if not isinstance(x, str) else x for x in v[:-1]
            )
            stop_loss_take_profit_tuple += (side,) + new_tuple

    objective = stop_loss_take_profit_wealth ** (1 / 12)

    params_dict = {
        "objective": objective,
        "yearly profit": stop_loss_take_profit_wealth,
        "params": args + stop_loss_take_profit_tuple,
    }

    return params_dict


# TODO: possible strategy types:
#   - stoch
#   - candlestick
#   - 2-timeframe ma
def find_optimal_aggregated_strategy(
    client,
    coin,
    bounds,
    resolutions,
    m,
    m_bear,
    frequency="high",
    strategy_type="ma",
    stop_loss_take_profit_types=["stop_loss", "take_profit", "trailing"],
    search_method="gradient",
    init_args=None,
    N_iter=None,
    N_repeat_inp=None,
    objective_dict=None,
    step=0.01,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    verbose=True,
    disable=False,
    short=True,
    Xs_index=[0, 1],
    taxes=False,
):

    parameter_names = get_parameter_names(strategy_type)

    if search_method not in ["gradient", "BayesOpt"]:
        raise ValueError("search_method")

    sides = ["long"]
    if short:
        assert m_bear > 0
        sides.append("short")

    files = glob.glob(f"data/{coin}/*.json")
    files.sort(key=get_time)

    Xs = load_all_data(files, Xs_index)

    if not isinstance(Xs, list):
        Xs = [Xs]

    N_years = max(len(X) for X in Xs) / minutes_in_a_year

    total_balance = get_total_balance(client, False)
    total_balance = max(total_balance, 1)

    # TODO: move this into a function
    potential_balances = np.logspace(
        np.log10(total_balance / (10 * N_years)),
        np.log10(total_balance * 100 * N_years),
        int(2000 * N_years),
    )
    potential_spreads = get_average_spread(coin, m, potential_balances, m_bear=m_bear)

    if short:
        X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    else:
        X_bears = [None] * len(Xs)
    X_origs = Xs
    if m > 1:
        Xs = [get_multiplied_X(X, m) for X in Xs]

    if search_method == "gradient":
        prev_args = None
        if init_args is None:
            init_args = ()
            for param_name in parameter_names:
                search_space = np.arange(
                    bounds[param_name][0],
                    bounds[param_name][1] + resolutions[param_name],
                    resolutions[param_name],
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
            ds = np.array(list(product([-1, 0, 1], repeat=len(bounds))))
            candidate_args = ds * resolution_values + args.reshape(1, -1)

            for i in range(len(bounds)):
                candidate_args[:, i] = np.clip(
                    candidate_args[:, i], *bounds[parameter_names[i]]
                )

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
                        coin,
                        m,
                        m_bear,
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
                        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
                        trail_value_recalc_period=trail_value_recalc_period,
                        workers=None,
                        debug=False,
                        taxes=taxes,
                    )

                if objective_dict[c_args]["objective"] > best_objective:
                    best_objective = objective_dict[c_args]["objective"]
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

    elif search_method == "BayesOpt":
        if verbose:
            print(
                "Total N_iter:",
                N_iter
                + np.sqrt(N_iter).astype(int)
                + (1 if init_args is not None else 0),
            )
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

        if strategy_type == "ma":

            def objective_fn(aggregate_N, w):
                args = (aggregate_N, w)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                        args,
                        strategy_type,
                        frequency,
                        coin,
                        m,
                        m_bear,
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
                        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
                        trail_value_recalc_period=trail_value_recalc_period,
                        workers=None,
                        debug=False,
                        taxes=taxes,
                    )

                return objective_dict[args]["objective"]

        elif strategy_type == "stoch":

            def objective_fn(aggregate_N, w, th):
                args = (aggregate_N, w, th)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                        args,
                        strategy_type,
                        frequency,
                        coin,
                        m,
                        m_bear,
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
                        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
                        trail_value_recalc_period=trail_value_recalc_period,
                        workers=None,
                        debug=False,
                        taxes=taxes,
                    )

                return objective_dict[args]["objective"]

        elif strategy_type == "macross":

            def objective_fn(aggregate_N, w, w2):
                args = (aggregate_N, w, w2)
                args = get_rounded_args(args, resolutions, parameter_names)

                if args not in objective_dict:
                    objective_dict[args] = get_objective_function(
                        args,
                        strategy_type,
                        frequency,
                        coin,
                        m,
                        m_bear,
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
                        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
                        trail_value_recalc_period=trail_value_recalc_period,
                        workers=None,
                        debug=False,
                        taxes=taxes,
                    )

                return objective_dict[args]["objective"]

        optimizer = BayesianOptimization(f=objective_fn, pbounds=bounds, verbose=2)

        if init_args is not None:
            optimizer.probe(
                params=dict(zip(parameter_names, init_args)),
                lazy=True,
            )

        optimizer.maximize(
            init_points=np.sqrt(N_iter).astype(int),
            n_iter=N_iter,
        )

        args = tuple(v for v in optimizer.max["params"].values())
        args = get_rounded_args(args, resolutions, parameter_names)

    return objective_dict[args], objective_dict


# TODO: same order of arguments as in find_optimal_aggregated_strategy
def save_optimal_parameters(
    all_bounds,
    all_resolutions,
    coins,
    frequencies,
    strategy_types,
    stop_loss_take_profit_types=["stop_loss", "take_profit", "trailing"],
    N_iter=None,
    m=3,
    m_bear=3,
    N_repeat_inp=40,
    step=0.01,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    skip_existing=False,
    verbose=True,
    disable=False,
    short=True,
    Xs_index=[0, 1],
    debug=False,
    taxes=False,
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
            print(coin, frequency, strategy_type, m, m_bear)

        options = [coin, frequency, strategy_type, str(m), str(m_bear)]
        parameter_names = get_parameter_names(strategy_type)

        fname = "optim_results/" + "_".join(options) + ".json"
        N_iter_default = 50

        if os.path.exists(fname):
            if skip_existing:
                if verbose:
                    print("Skipping...")
                    print()
                continue

            with open(fname, "r") as file:
                params_dict = json.load(file)
                init_args = params_dict["params"][: len(parameter_names)]

            modification_time = os.path.getmtime(fname)

            N_iter1 = (time.time() - modification_time) // (60 * 60 * 24)
            N_iter1 = min(N_iter1, N_iter_default)
        else:
            init_args = None
            N_iter1 = N_iter_default

        bounds = dict([(k, all_bounds[k]) for k in parameter_names])
        resolutions = dict([(k, all_resolutions[k]) for k in parameter_names])

        if N_iter is not None:
            N_iter1 = N_iter

        if len(bounds) > 2:
            N_iter1 *= 2 ** (len(bounds) - 2)

        N_iter1 = max(N_iter1, 1)

        if verbose:
            print("Bayesian Optimization...")

        params_dict_candidate, objective_dict = find_optimal_aggregated_strategy(
            client=client,
            coin=coin,
            bounds=bounds,
            resolutions=resolutions,
            m=m,
            m_bear=m_bear,
            frequency=frequency,
            strategy_type=strategy_type,
            stop_loss_take_profit_types=stop_loss_take_profit_types,
            search_method="BayesOpt",
            init_args=init_args,
            N_iter=N_iter1,
            N_repeat_inp=N_repeat_inp,
            objective_dict=None,
            step=step,
            place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
            trail_value_recalc_period=trail_value_recalc_period,
            verbose=verbose,
            disable=disable,
            short=short,
            Xs_index=Xs_index,
            taxes=taxes,
        )

        if verbose:
            print("Gradient Descent...")

        params_dict_candidate, _ = find_optimal_aggregated_strategy(
            client=client,
            coin=coin,
            bounds=bounds,
            resolutions=resolutions,
            m=m,
            m_bear=m_bear,
            frequency=frequency,
            strategy_type=strategy_type,
            stop_loss_take_profit_types=stop_loss_take_profit_types,
            search_method="gradient",
            init_args=params_dict_candidate["params"][: len(bounds)],
            N_iter=N_iter,
            N_repeat_inp=N_repeat_inp,
            objective_dict=objective_dict,
            step=step,
            place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
            trail_value_recalc_period=trail_value_recalc_period,
            verbose=verbose,
            disable=disable,
            short=short,
            Xs_index=Xs_index,
            taxes=taxes,
        )

        if not debug:
            are_same = True
            if init_args is not None:
                if len(params_dict_candidate["params"]) != len(params_dict["params"]):
                    are_same = False

                if are_same:
                    for i in range(len(params_dict_candidate["params"])):
                        if (
                            params_dict_candidate["params"][i]
                            != params_dict["params"][i]
                        ):
                            are_same = False
                            break

            if are_same and init_args is not None:
                if verbose:
                    print(
                        "Optimal parameters did not change for:",
                        coin,
                        frequency,
                        strategy_type,
                        m,
                        m_bear,
                    )
            else:
                params_dict = params_dict_candidate
                with open("optim_results/" + "_".join(options) + ".json", "w") as file:
                    json.dump(params_dict, file, cls=NpEncoder)

        if verbose:
            print()
            print_dict(params_dict)
            print()


def get_adaptive_wealths_for_multiple_strategies(
    strategies,
    weights,
    N_repeat_inp=60,
    compress=60,  # 1440,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    disable=False,
    verbose=True,
    randomize=False,
    Xs_index=[0, 1],
    debug=False,
    taxes=False,
):

    # TODO: have improved logic on this (i.e. a new boolean variable telling what to do)
    # if trail_value_recalc_period is None:
    #     trail_value_recalc_period = get_average_trading_period(strategies)

    client = FtxClient(ftx_api_key, ftx_secret_key)

    aggregate_Ns = [v["params"][0] for v in strategies.values()]
    ws = [v["params"][1] for v in strategies.values()]

    max_aggregate_N_times_w = np.max(np.prod(list(zip(aggregate_Ns, ws)), axis=1))

    min_aggregate_N = np.min(aggregate_Ns)
    max_aggregate_N = np.max(aggregate_Ns)

    if N_repeat_inp is None:
        N_repeat = min_aggregate_N * 60
    else:
        N_repeat = N_repeat_inp

    N_repeat = min(N_repeat, min_aggregate_N * 60)

    if randomize:
        rand_Ns = np.random.choice(
            np.arange(0, min_aggregate_N * 60), size=N_repeat, replace=True
        )
    else:
        rand_Ns = np.arange(0, min_aggregate_N * 60, min_aggregate_N * 60 // N_repeat)

    Xs_dict = {}
    times_dict = {}
    wealths_dict = {}

    for strategy_key in strategies.keys():
        coin = strategy_key.split("_")[0]
        if coin not in Xs_dict:
            files = glob.glob(f"data/{coin}/*.json")
            files.sort(key=get_time)
            Xs, times = load_all_data(files, Xs_index, True)

            if not isinstance(Xs, list):
                Xs = [Xs]
                times = [times]

            Xs_dict[coin] = Xs
            times_dict[coin] = times

    if not isinstance(Xs_index, list):
        Xs_index = [Xs_index]

    # make the starting and ending times the same
    for i in range(len(Xs_index)):
        min_time = np.min([v[i] for v in times_dict.values()])
        max_time = np.max([v[i] for v in times_dict.values()])

        for coin in Xs_dict.keys():
            time_diff_from_min = (times_dict[coin][i] - min_time) // 60
            time_diff_to_max = (max_time - times_dict[coin][i]) // 60

            if time_diff_from_min > 0:
                Xs_dict[coin][i] = Xs_dict[coin][i][:-time_diff_from_min]

            if time_diff_to_max > 0:
                Xs_dict[coin][i] = Xs_dict[coin][i][time_diff_to_max:]

    for i in range(len(Xs_index)):
        assert len(np.unique([len(Xs[i]) for Xs in Xs_dict.values()])) == 1

    for strategy_key, strategy in strategies.items():
        strategy_type = strategy_key.split("_")[2]
        parameter_names = get_parameter_names(strategy_type)

        m, m_bear = tuple([int(x) for x in strategy_key.split("_")[3:]])

        args = strategy["params"][: len(parameter_names)]
        aggregate_N, w = args[:2]

        sltp_args = strategy["params"][len(parameter_names) :]

        if place_take_profit_and_stop_loss_simultaneously:
            short = "short" in sltp_args
        else:
            short = len(sltp_args) == 4

        get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

        coin = strategy_key.split("_")[0]
        Xs = Xs_dict[coin]

        N_years = max(len(X) for X in Xs) / minutes_in_a_year

        total_balance = get_total_balance(client, False) * weights[strategy_key]
        total_balance = max(total_balance, 1)

        potential_balances = np.logspace(
            np.log10(total_balance / (10 * N_years)),
            np.log10(total_balance * 100 * N_years),
            int(2000 * N_years),
        )
        potential_spreads = get_average_spread(
            coin, m, potential_balances, m_bear=m_bear
        )

        if short:
            X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
        else:
            X_bears = [None] * len(Xs)
        X_origs = Xs
        if m > 1:
            Xs = [get_multiplied_X(X, m) for X in Xs]

        n_months = 0

        if verbose:
            print(strategy_key, strategy["objective"], strategy["params"])

        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            min_N_orig = len(X_orig) - max_aggregate_N_times_w * 60

            buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(
                X_orig, *args, from_minute=True
            )

            pbar = tqdm(rand_Ns, disable=disable)
            for rand_N in pbar:
                min_N = np.min(
                    [
                        ((min_N_orig - rand_N) // (agg_N * 60)) * agg_N * 60
                        - (agg_N * 60 - 1)
                        for agg_N in aggregate_Ns
                    ]
                )
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

                buys_candidate = np.append(
                    np.repeat(buys[:-1], aggregate_N * 60), buys[-1]
                )
                sells_candidate = np.append(
                    np.repeat(sells[:-1], aggregate_N * 60), sells[-1]
                )

                buys = buys_candidate[-min_N:]
                sells = sells_candidate[-min_N:]

                wealths = get_adaptive_wealths(
                    X=X1[-min_N:, :],
                    buys=buys,
                    sells=sells,
                    aggregate_N=aggregate_N,
                    sltp_args=sltp_args,
                    total_balance=total_balance,
                    potential_balances=potential_balances,
                    potential_spreads=potential_spreads,
                    place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
                    trail_value_recalc_period=trail_value_recalc_period,
                    commissions=commissions,
                    short=short,
                    X_bear=X1_bear[-min_N:, :] if short else None,
                    taxes=taxes,
                )

                if debug:
                    print(wealths[-1])
                    quick_plot(
                        xs=[X1[-min_N:, 0], wealths],
                        colors=["k", "g"],
                        alphas=[0.5, 0.8],
                        log=True,
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
    weights,
    N_repeat_inp=60,
    compress=60,  # 1440,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    disable=False,
    verbose=True,
    randomize=False,
    Xs_index=[0, 1],
    debug=False,
    taxes=False,
):

    df = get_adaptive_wealths_for_multiple_strategies(
        strategies=strategies,
        weights=weights,
        N_repeat_inp=N_repeat_inp,
        compress=compress,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=trail_value_recalc_period,
        disable=disable,
        verbose=verbose,
        randomize=randomize,
        Xs_index=Xs_index,
        debug=False,
        taxes=taxes,
    )

    # TODO: have this as a input parameter
    df.to_csv(f"optim_results/wealths_{compress}.csv")


def optimize_weights(compress, save=True, verbose=False):
    df = pd.read_csv(f"optim_results/wealths_{compress}.csv", index_col=0)

    freq = minutes_in_a_year / compress
    mu = expected_returns.mean_historical_return(df, frequency=freq)
    S = risk_models.CovarianceShrinkage(df, frequency=freq).ledoit_wolf()
    # if verbose:
    #     print(mu)
    #     print(S)

    bl = BlackLittermanModel(S, pi=np.zeros((len(df.columns),)), absolute_views=mu)
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

        X_change = X[1:, :] / X[:-1, :]

        wealths = np.prod(X_change, axis=0) ** (freq / len(X))
        # print(wealths)

        weighted_diffs = np.matmul(X_change, weights)
        wealth = np.prod(weighted_diffs) ** (freq / len(X))
        print(wealth)

    if save:
        with open("optim_results/weights.json", "w") as file:
            json.dump(weights_dict, file)

    return weights_dict, a, b, c


def optimize_weights_iterative(
    coins=["ETH", "BTC"],
    freqs=["low", "high"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    n_iter=20,
    compress=None,
    place_take_profit_and_stop_loss_simultaneously=False,
    init_trail_value_recalc_period=None,
    taxes=False,
):

    strategies, weights = get_filtered_strategies_and_weights(
        coins=coins,
        freqs=freqs,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        filter=False,
    )

    if compress is None:
        compress = get_average_trading_period(strategies)

    print(f"Compress: {compress}")

    # TODO: improved logic
    # if init_trail_value_recalc_period is None:
    #     trail_value_recalc_period = get_average_trading_period(strategies)
    #     print(trail_value_recalc_period)
    # else:
    #     trail_value_recalc_period = init_trail_value_recalc_period
    trail_value_recalc_period = init_trail_value_recalc_period

    weight_values = np.zeros((len(strategies),))

    for n in range(n_iter):

        weight_values *= n
        weight_values += np.array(list(weights.values()))
        weight_values /= n + 1
        print(weight_values)
        print()
        weights = dict(list(zip(weights.keys(), weight_values)))

        save_wealths(
            strategies,
            weights=weights,
            N_repeat_inp=5,
            compress=compress,
            place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
            trail_value_recalc_period=trail_value_recalc_period,
            disable=True,
            verbose=False,
            randomize=False,
            Xs_index=[0, 1],
            debug=False,
            taxes=taxes,
        )

        weights, a, b, c = optimize_weights(compress, save=True, verbose=False)

        # would_be_strategies, _ = get_filtered_strategies_and_weights(
        #     coins = coins,
        #     freqs = freqs,
        #     strategy_types = strategy_types,
        #     ms = ms,
        #     m_bears = m_bears,
        #     filter = True
        # )
        # trail_value_recalc_period = get_average_trading_period(would_be_strategies)
        # print(trail_value_recalc_period)

        print("Weights:")
        print_dict(weights)
        print("Performance:")
        print(a, b, c)
        print()

    # TODO: delete 'wealths.csv'


# TODO: optimize sharpe ratio instead of improvement
def get_optimal_balancing_period(
    coins=["ETH", "BTC"],
    freqs=["low", "high"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    N_repeat=1,
    w=11,
    max_balancing_period_in_days=7,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    randomize=True,
    Xs_index=[0, 1],
    plot=False,
    taxes=False,
):

    strategies, weights = get_filtered_strategies_and_weights(
        coins, freqs, strategy_types, ms, m_bears
    )

    compress = 60

    # TODO: improved logic
    # if trail_value_recalc_period is None:
    #     trail_value_recalc_period = get_average_trading_period(strategies)

    keys = list(weights.keys())
    weight_values = np.array(list(weights.values()))

    client = FtxClient(ftx_api_key, ftx_secret_key)

    total_balance = get_total_balance(client, False) * np.max(weight_values)
    total_balance = max(total_balance, 1)

    fnames = glob.glob(f"data/{coins[0]}/*.json")
    N_years = len(fnames) * 2000 / minutes_in_a_year
    # print("N_years:", round_to_n(N_years, 4))
    # print()

    potential_balances = np.logspace(
        np.log10(total_balance / (10 * N_years)),
        np.log10(total_balance * 100 * N_years),
        int(2000 * N_years),
    )
    potential_spreads = {}

    for coin, m, m_bear in product(coins, ms, m_bears):
        potential_spreads[(coin, m, m_bear)] = get_average_spread(
            coin, m, potential_balances, m_bear=m_bear
        )
        # print(coin, m, m_bear)

    balancing_periods = np.arange(1, 24 * max_balancing_period_in_days + 1)
    balanced_improvements = np.zeros((N_repeat, len(balancing_periods)))

    for i in range(N_repeat):
        df = get_adaptive_wealths_for_multiple_strategies(
            strategies,
            weights,
            N_repeat_inp=1,
            compress=compress,
            place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
            trail_value_recalc_period=trail_value_recalc_period,
            disable=True,
            verbose=False,
            randomize=randomize,
            Xs_index=Xs_index,
            taxes=taxes,
        )

        wealths = (df.iloc[-1] / df.iloc[0]) ** (
            minutes_in_a_year / (len(df) * compress)
        )

        X = df.values
        n = X.shape[0]

        weighted_wealths = np.matmul(X, weight_values)
        wealth = (weighted_wealths[-1] / weighted_wealths[0]) ** (
            minutes_in_a_year / (len(df) * compress)
        )
        print("Non-balanced wealth (yearly):", round_to_n(wealth, 4))

        for b in balancing_periods:
            balanced_wealths = get_balanced_wealths(
                X,
                weight_values,
                keys,
                potential_spreads,
                potential_balances,
                total_balance,
                balancing_period=b,
                taxes=taxes,
            )
            assert n == X.shape[0]

            balanced_wealth = balanced_wealths[-1] ** (
                minutes_in_a_year / (len(df) * compress)
            )
            improvement = balanced_wealth / wealth
            # print("Period:", b,
            #     "Balanced wealth (yearly):", round_to_n(balanced_wealth, 4),
            #     "Improvement:", round_to_n(improvement, 4))

            balanced_improvements[i, b - 1] = improvement

    balanced_improvements = np.exp(np.mean(np.log(balanced_improvements), axis=0))
    assert balanced_improvements.shape[0] == balancing_periods.shape[0]

    avg_balanced_improvements = sma(balanced_improvements, w).flatten()

    balancing_period = np.argmax(avg_balanced_improvements) + w // 2 + 1
    # balancing_period1 = np.argmax(balanced_improvements) + 1

    print()
    print("Best balancing period:", balancing_period)
    print(
        "Average improvement:", avg_balanced_improvements[balancing_period - w // 2 - 1]
    )
    print()

    if plot:
        plt.style.use("seaborn")
        plt.plot(balancing_periods, balanced_improvements)
        plt.plot(
            np.arange(len(avg_balanced_improvements)) + w // 2 + 1,
            avg_balanced_improvements,
        )
        plt.show()

    return balancing_period


def plot_weighted_adaptive_wealths(
    coins=["ETH", "BTC"],
    freqs=["low", "high"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    N_repeat=1,
    compress=None,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    randomize=True,
    Xs_index=[0, 1],
    active_balancing=False,
    balancing_period=1,
    taxes=False,
):

    plt.style.use("seaborn")

    strategies, weights = get_filtered_strategies_and_weights(
        coins, freqs, strategy_types, ms, m_bears
    )
    print_dict(strategies)

    if compress is None:
        compress = get_average_trading_period(strategies)
        print("Compress:", compress)

    # TODO: improved logic
    # if trail_value_recalc_period is None:
    #     trail_value_recalc_period = get_average_trading_period(strategies)

    keys = list(weights.keys())
    weight_values = np.array(list(weights.values()))

    client = FtxClient(ftx_api_key, ftx_secret_key)

    total_balance = get_total_balance(client, False) * np.max(weight_values)
    total_balance = max(total_balance, 1)

    fnames = glob.glob(f"data/{coins[0]}/*.json")
    N_years = len(fnames) * 2000 / minutes_in_a_year
    print("N_years:", round_to_n(N_years, 4))
    print()

    potential_balances = np.logspace(
        np.log10(total_balance / (10 * N_years)),
        np.log10(total_balance * 100 * N_years),
        int(2000 * N_years),
    )
    potential_spreads = {}

    for coin, m, m_bear in product(coins, ms, m_bears):
        potential_spreads[(coin, m, m_bear)] = get_average_spread(
            coin, m, potential_balances, m_bear=m_bear
        )
        # print(coin, m, m_bear)

    # TODO: move theis loop into get_adaptive_wealths_for_multiple_strategies?
    for i in range(N_repeat):
        df = get_adaptive_wealths_for_multiple_strategies(
            strategies,
            weights,
            N_repeat_inp=1,
            compress=compress,
            place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
            trail_value_recalc_period=trail_value_recalc_period,
            disable=True,
            verbose=False,
            randomize=randomize,
            Xs_index=Xs_index,
            taxes=taxes,
        )

        wealths = (df.iloc[-1] / df.iloc[0]) ** (
            minutes_in_a_year / (len(df) * compress)
        )

        X = df.values

        weighted_wealths = np.matmul(X, weight_values)
        wealth = (weighted_wealths[-1] / weighted_wealths[0]) ** (
            minutes_in_a_year / (len(df) * compress)
        )
        print("Non-balanced wealth (yearly):", round_to_n(wealth, 4))

        plt.plot(X, "k", alpha=0.25 / np.sqrt(N_repeat))
        plt.plot(weighted_wealths, "b", alpha=1.0 / np.sqrt(N_repeat))

        if active_balancing:
            balanced_wealths = get_balanced_wealths(
                X,
                weight_values,
                keys,
                potential_spreads,
                potential_balances,
                total_balance,
                balancing_period=balancing_period,
                taxes=taxes,
            )

            balanced_wealth = balanced_wealths[-1] ** (
                minutes_in_a_year / (len(df) * compress)
            )
            print("Balanced wealth (yearly):", round_to_n(balanced_wealth, 4))
            print("Improvement:", round_to_n(balanced_wealth / wealth, 4))

            plt.plot(
                np.linspace(0, len(weighted_wealths) - 1, len(balanced_wealths)),
                balanced_wealths,
                "g",
                alpha=1.0 / np.sqrt(N_repeat),
            )

        print()

    plt.yscale("log")
    plt.show()


def get_displacements_and_last_signal_change_time(
    coins=["ETH", "BTC"],
    frequencies=["low", "high"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    sep=2,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    Xs_index=[0, 1],
    plot=True,
    verbose=True,
    disable=False,
    taxes=False,
):

    plt.style.use("seaborn")
    client = FtxClient(ftx_api_key, ftx_secret_key)

    # as long as cryptocompare is used in trade.py, displacements can only be
    # calculateed for frequency == 'high' strategies
    strategies, weights = get_filtered_strategies_and_weights(
        coins, frequencies, strategy_types, ms, m_bears, normalize=False
    )

    if len(strategies) == 0:
        return {}

    # TODO: improved logic
    # if trail_value_recalc_period is None:
    #     trail_value_recalc_period = get_average_trading_period(strategies)

    trade_wealth_categories = ["base", "min", "max"]

    flag = True

    Xs_dict = {}
    saved_times_dict = {}
    online_times_dict = {}
    wealths_dict = {}
    last_buy_signal = {}
    last_signal_change_times = {}
    triggered = {}

    for strategy_key, strategy in strategies.items():
        if verbose:
            print(strategy_key, strategy["objective"], strategy["params"])

        strategy_type = strategy_key.split("_")[2]
        frequency = strategy_key.split("_")[1]
        m, m_bear = tuple([int(x) for x in strategy_key.split("_")[3:]])

        parameter_names = get_parameter_names(strategy_type)

        args = strategy["params"][: len(parameter_names)]
        aggregate_N = args[0]

        sltp_args = strategy["params"][len(parameter_names) :]
        if place_take_profit_and_stop_loss_simultaneously:
            short = "short" in sltp_args
        else:
            short = len(sltp_args) == 4

        sides = ["long"]
        if short:
            sides.append("short")

        get_buys_and_sells_fn = choose_get_buys_and_sells_fn(strategy_type)

        coin = strategy_key.split("_")[0]
        if coin not in Xs_dict:
            files = glob.glob(f"data/{coin}/*.json")
            files.sort(key=get_time)
            Xs, saved_times = load_all_data(files, Xs_index, True)
            _, online_time = get_recent_data(coin, 10, "h", 1)

            if not isinstance(Xs, list):
                Xs = [Xs]
                saved_times = [saved_times]

            Xs_dict[coin] = Xs
            saved_times_dict[coin] = saved_times
            online_times_dict[coin] = online_time

        Xs = Xs_dict[coin]
        saved_times = saved_times_dict[coin]
        online_time = online_times_dict[coin]

        N_years = max(len(X) for X in Xs) / minutes_in_a_year
        print("N_years:", N_years)

        total_balance = get_total_balance(client, False) * weights[strategy_key]
        total_balance = max(total_balance, 1)

        potential_balances = np.logspace(
            np.log10(total_balance / (10 * N_years)),
            np.log10(total_balance * 100 * N_years),
            int(2000 * N_years),
        )
        potential_spreads = get_average_spread(
            coin, m, potential_balances, m_bear=m_bear
        )

        # TODO: move this into a function
        if short:
            X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
        else:
            X_bears = [None] * len(Xs)
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

        last_signal_change_times[strategy_key] = -1

        for i in range(len(Xs)):
            X_orig, X, X_bear = X_origs[i], Xs[i], X_bears[i]
            buys_orig, sells_orig, N_orig = get_buys_and_sells_fn(
                X_orig, *args, from_minute=True
            )

            if i == len(Xs) - 1:
                last_buy_signal[strategy_key] = buys_orig[-1] > sells_orig[-1]

            last_signal_change_i = (
                np.argwhere(buys_orig != buys_orig[-1]).flatten()[-1] + 1
            )
            last_signal_change_time_candidate = (
                saved_times[i] - (N_orig - 1 - last_signal_change_i) * 60
            )

            # print(np.argwhere(buys_orig != buys_orig[-1]).flatten())
            # print(i, last_signal_change_i, last_signal_change_time_candidate, \
            #     (saved_times[i] - last_signal_change_time_candidate) / (60 * 60 * 24), \
            #     buys_orig[-1] > sells_orig[-1])

            if (
                last_signal_change_time_candidate
                > last_signal_change_times[strategy_key]
            ):
                last_signal_change_times[
                    strategy_key
                ] = last_signal_change_time_candidate

                if buys_orig[-1] > sells_orig[-1]:
                    if place_take_profit_and_stop_loss_simultaneously:
                        stop_loss = sltp_args[3]
                        stop_loss_type_is_trailing = sltp_args[2] == "trailing"
                        take_profit = sltp_args[1]
                    else:
                        if sltp_args[0] == "take_profit":
                            stop_loss = 0
                            take_profit = sltp_args[1]
                        else:
                            stop_loss = sltp_args[1]
                            take_profit = np.Inf

                        stop_loss_type_is_trailing = sltp_args[0] == "trailing"

                    triggered[strategy_key] = last_signal_change_i < N_orig - 1

                    if triggered[strategy_key]:
                        (
                            _,
                            triggered[strategy_key],
                        ) = get_stop_loss_take_profit_wealths_from_sub_trades(
                            [
                                X[last_signal_change_i + 1 :, 0]
                                / X[last_signal_change_i, 0]
                            ],
                            [
                                X[last_signal_change_i + 1 :, 2]
                                / X[last_signal_change_i, 0]
                            ],
                            [
                                X[last_signal_change_i + 1 :, 1]
                                / X[last_signal_change_i, 0]
                            ],
                            stop_loss,
                            stop_loss_type_is_trailing,
                            take_profit,
                            trail_value_recalc_period=trail_value_recalc_period,
                            return_latest_trigger_status=True,
                            taxes=taxes,
                        )
                elif short:
                    if place_take_profit_and_stop_loss_simultaneously:
                        stop_loss = sltp_args[7]
                        stop_loss_type_is_trailing = sltp_args[6] == "trailing"
                        take_profit = sltp_args[5]
                    else:
                        if sltp_args[2] == "take_profit":
                            stop_loss = 0
                            take_profit = sltp_args[3]
                        else:
                            stop_loss = sltp_args[3]
                            take_profit = np.Inf

                        stop_loss_type_is_trailing = sltp_args[2] == "trailing"

                    triggered[strategy_key] = last_signal_change_i < N_orig - 1

                    if triggered[strategy_key]:
                        (
                            _,
                            triggered[strategy_key],
                        ) = get_stop_loss_take_profit_wealths_from_sub_trades(
                            [
                                X_bear[last_signal_change_i + 1 :, 0]
                                / X_bear[last_signal_change_i, 0]
                            ],
                            [
                                X_bear[last_signal_change_i + 1 :, 2]
                                / X_bear[last_signal_change_i, 0]
                            ],
                            [
                                X_bear[last_signal_change_i + 1 :, 1]
                                / X_bear[last_signal_change_i, 0]
                            ],
                            stop_loss,
                            stop_loss_type_is_trailing,
                            take_profit,
                            trail_value_recalc_period=trail_value_recalc_period,
                            return_latest_trigger_status=True,
                            taxes=taxes,
                        )

            if frequency == "low":
                continue

            wealth_list = []
            time_diff = ((online_time - saved_times[i]) // 60) % (aggregate_N * 60)

            for rand_N in tqdm(range(aggregate_N * 60), disable=disable):
                (
                    trade_wealths_dict,
                    sub_trade_wealths_dict,
                    N_transactions_buy,
                    N_transactions_sell,
                    total_months,
                ) = get_trade_wealths_dicts(
                    rand_N,
                    X,
                    X_bear,
                    short,
                    aggregate_N,
                    buys_orig,
                    sells_orig,
                    N_orig,
                    sides,
                    trade_wealth_categories,
                )

                stop_loss_take_profit_wealths_dict = {}

                for i, side in enumerate(sides):
                    if place_take_profit_and_stop_loss_simultaneously:
                        take_profit = sltp_args[i * 4 + 1]
                        stop_loss_type = sltp_args[i * 4 + 2]
                        stop_loss = sltp_args[i * 4 + 3]

                        wealth = get_stop_loss_take_profit_wealths_from_sub_trades(
                            sub_trade_wealths_dict[side]["base"],
                            sub_trade_wealths_dict[side]["min"],
                            sub_trade_wealths_dict[side]["max"],
                            stop_loss=stop_loss,
                            stop_loss_type_is_trailing=stop_loss_type == "trailing",
                            take_profit=take_profit,
                            total_months=total_months,
                            trail_value_recalc_period=trail_value_recalc_period,
                            return_total=True,
                            return_as_log=False,
                            taxes=taxes,
                        )

                        stop_loss_take_profit_wealths_dict[side] = (
                            side,
                            take_profit,
                            stop_loss_type,
                            stop_loss,
                            wealth,
                        )

                    else:
                        # TODO: not working, fix/update to use current version
                        sltp_type = sltp_args[i * 2]
                        sltp_param = sltp_args[i * 2 + 1]

                        wealth_log = get_stop_loss_take_profit_wealth(
                            side,
                            sltp_type,
                            trade_wealths_dict,
                            sub_trade_wealths_dict,
                            [sltp_param],
                            [sltp_param],
                            trail_value_recalc_period=trail_value_recalc_period,
                        )
                        wealth_log = wealth_log[0]

                        wealth = apply_commissions_and_spreads(
                            wealth_log,
                            0,
                            0.0,
                            0.0,
                            n_months=total_months,
                            from_log=True,
                        )

                        stop_loss_take_profit_wealths_dict[side] = (
                            sltp_type,
                            sltp_param,
                            wealth,
                        )

                wealth = get_semi_adaptive_wealths(
                    total_balance,
                    potential_balances,
                    potential_spreads,
                    stop_loss_take_profit_wealths_dict,
                    N_transactions_buy,
                    N_transactions_sell,
                    total_months,
                    debug=False,
                )
                wealth = wealth ** (1 / 12)

                wealth_list.append(wealth)

            wealth_list = np.flip(np.array(wealth_list))
            wealth_list = np.roll(wealth_list, -time_diff + 1)

            new_wealth_list = np.ones((60,))

            for n in range(aggregate_N):
                new_wealth_list *= wealth_list[n * 60 : (n + 1) * 60]

            wealth_list = new_wealth_list ** (1 / aggregate_N)

            wealth_lists.append(wealth_list)

        if frequency == "low":
            continue

        wealth_lists = np.stack(wealth_lists)
        wealth_list = np.exp(np.sum(np.log(wealth_lists) * len_weights, axis=0))
        wealth_i = np.argmax(wealth_list)
        if verbose:
            print(wealth_i, wealth_list[wealth_i], wealth_list[wealth_i] ** 12)
            print()

        wealths_dict[strategy_key] = wealth_list

    diff = np.arange(-sep + 1, sep)

    li = np.ones((60,)).astype(bool)
    li[diff % 60] = False

    idx = np.arange(60)
    not_processed = np.ones((len(wealths_dict),)).astype(bool)
    keys = np.array(list(wealths_dict.keys()))
    keys_idx = np.arange(len(wealths_dict))

    def _helper(key, li):
        _idx = idx[li]
        wealth_i = np.argmax(wealths_dict[key][li])
        return _idx[wealth_i], wealths_dict[key][li][wealth_i]

    displacements = {k: (0, 1.0) for k in strategies.keys() if k not in wealths_dict}

    for i in range(len(wealths_dict)):
        if not np.any(li):
            raise ValueError(
                f"No valid displacements available (step {i+1}/{len(wealths_dict)}), 'sep' must be decreased."
            )

        max_i = np.argmax([_helper(key, li)[1] for key in keys[not_processed]])
        max_i = keys_idx[not_processed][max_i]
        not_processed[max_i] = False
        key = keys[max_i]
        displacements[key] = _helper(key, li)

        li[(diff + displacements[key][0]) % 60] = False

    res = {}

    for key in strategies.keys():
        res[key] = displacements[key] + (
            last_buy_signal[strategy_key],
            last_signal_change_times[key],
            triggered[key],
        )

    if verbose:
        print_dict(res)

    if plot and wealths_dict:
        for key in wealths_dict.keys():
            plt.plot(wealths_dict[key])
        plt.legend(list(wealths_dict.keys()))
        for key in wealths_dict.keys():
            i, w = displacements[key]
            plt.plot([i], [w], "k.", alpha=0.8)
        plt.show()

    return res


def save_displacements_and_last_signal_change_time(
    coins=["ETH", "BTC"],
    strategy_types=["ma", "macross"],
    ms=[1, 3],
    m_bears=[0, 3],
    sep=2,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    Xs_index=[0, 1],
    plot=True,
    verbose=True,
    disable=False,
    taxes=False,
):

    displacements_and_last_signal_change_time = get_displacements_and_last_signal_change_time(
        coins=coins,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        sep=sep,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=trail_value_recalc_period,
        Xs_index=Xs_index,
        plot=plot,
        verbose=verbose,
        disable=disable,
        taxes=taxes,
    )

    with open(
        "optim_results/displacements_and_last_signal_change_time.json", "w"
    ) as file:
        json.dump(displacements_and_last_signal_change_time, file, cls=NpEncoder)


if __name__ == "__main__":
    all_bounds = {
        "aggregate_N": (2, 12),
        "w": (1, 60),
        "w2": (1, 25),
    }
    all_resolutions = {
        "aggregate_N": 1,
        "w": 1,
        "w2": 1,
    }
    coins = ["ETH", "BTC"]
    frequencies = ["low", "high"]
    strategy_types = ["ma", "macross"]
    stop_loss_take_profit_types = ["stop_loss", "take_profit", "trailing"]
    N_iter = None
    m = 3
    m_bear = 3
    N_repeat_inp = 10
    step = 0.01
    place_take_profit_and_stop_loss_simultaneously = True
    trail_value_recalc_period = 200
    skip_existing = False
    verbose = True
    disable = False
    short = True
    Xs_index = [0, 1]
    debug = False
    active_balancing = True
    taxes = False

    # save_optimal_parameters(
    #     all_bounds = all_bounds,
    #     all_resolutions = all_resolutions,
    #     coins = coins,
    #     frequencies = frequencies,
    #     strategy_types = strategy_types,
    #     stop_loss_take_profit_types = stop_loss_take_profit_types,
    #     N_iter = N_iter,
    #     m = m,
    #     m_bear = m_bear,
    #     N_repeat_inp = N_repeat_inp,
    #     step = step,
    #     place_take_profit_and_stop_loss_simultaneously = place_take_profit_and_stop_loss_simultaneously,
    #     trail_value_recalc_period = trail_value_recalc_period,
    #     skip_existing = skip_existing,
    #     verbose = verbose,
    #     disable = disable,
    #     short = short,
    #     Xs_index = Xs_index,
    #     debug = debug,
    #     taxes = taxes
    # )

    save_optimal_parameters(
        all_bounds=all_bounds,
        all_resolutions=all_resolutions,
        coins=coins,
        frequencies=frequencies,  # ['low'],
        strategy_types=strategy_types,
        stop_loss_take_profit_types=stop_loss_take_profit_types,
        N_iter=N_iter,
        m=1,
        m_bear=0,
        N_repeat_inp=N_repeat_inp,
        step=step,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=trail_value_recalc_period,
        skip_existing=skip_existing,
        verbose=verbose,
        disable=disable,
        short=False,
        Xs_index=Xs_index,
        debug=debug,
        taxes=taxes,
    )

    n_iter = 10
    compress = 1440
    ms = [1]
    m_bears = [0]
    # ms = [1, 3]
    # m_bears = [0, 1, 3]

    optimize_weights_iterative(
        coins=coins,
        freqs=frequencies,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        n_iter=n_iter,
        compress=compress,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        init_trail_value_recalc_period=trail_value_recalc_period,
        taxes=taxes,
    )

    sep = 2
    plot = True

    save_displacements_and_last_signal_change_time(
        coins=coins,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        sep=sep,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=trail_value_recalc_period,
        Xs_index=Xs_index,
        plot=plot,
        verbose=verbose,
        disable=disable,
        taxes=taxes,
    )

    randomize = True
    N_repeat = 5
    w = 11

    balancing_period = get_optimal_balancing_period(
        coins=coins,
        freqs=frequencies,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        N_repeat=N_repeat,
        w=w,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=trail_value_recalc_period,
        randomize=randomize,
        Xs_index=Xs_index,
        plot=True,
        taxes=taxes,
    )

    N_repeat = 3

    plot_weighted_adaptive_wealths(
        coins=coins,
        freqs=frequencies,
        strategy_types=strategy_types,
        ms=ms,
        m_bears=m_bears,
        N_repeat=N_repeat,
        compress=60 * balancing_period,
        place_take_profit_and_stop_loss_simultaneously=place_take_profit_and_stop_loss_simultaneously,
        trail_value_recalc_period=60 * balancing_period,
        randomize=randomize,
        Xs_index=Xs_index,
        active_balancing=active_balancing,
        balancing_period=1,
        taxes=taxes,
    )
