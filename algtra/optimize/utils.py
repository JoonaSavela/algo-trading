import os
import sys

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from algtra.collect import data
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from algtra import utils
from tqdm import tqdm
import copy
from algtra import constants
import keys
from ftx.rest.client import FtxClient
from datetime import date, timedelta, datetime
from ciso8601 import parse_datetime
from functools import reduce
import glob
from numba import njit
import json
from bayes_opt import BayesianOptimization


def get_symbols_for_optimization(k=None, min_length=None, optim_results_dir="optim_results", verbose=False):
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    volumes_fname = os.path.abspath(os.path.join(data_dir, "volumes.json"))

    with open(volumes_fname, "r") as file:
        volumes = pd.Series(json.load(file))

    if min_length is not None:
        data_lengths = volumes.index.map(
            lambda x: data.load_price_data(
                data_dir, x + "/USD", return_price_data=False
            )[1]
        )

        volumes = volumes[data_lengths >= min_length]

    whitelist_fname = os.path.join(optim_results_dir, "whitelist.json")
    with open(whitelist_fname, "r") as file:
        whitelist = json.load(file)

    li = volumes.index.map(
        lambda x: whitelist[x]
    )
    volumes = volumes[li]

    volumes = volumes.sort_values(ascending=False)

    if k is not None:
        volumes = volumes.head(k)

    if verbose:
        print(volumes)
        print()

    return volumes.index.values


def get_N_iter_and_init_args(fname, n_params, max_N_iter=50):
    if os.path.exists(fname):
        with open(fname, "r") as file:
            params_dict = json.load(file)
            init_args = params_dict["params"]

        modification_time = os.path.getmtime(fname)

        N_iter = (time.time() - modification_time) // (60 * 60 * 24)
        N_iter = min(N_iter, max_N_iter)
    else:
        init_args = None
        N_iter = max_N_iter

    # Scale N_iter exponentially by dimensionality
    if n_params > 2:
        N_iter *= 2 ** (n_params - 2)

    N_iter = max(N_iter, 1)

    return N_iter, init_args


def get_rounded_args(args, resolutions, parameter_names):
    res = ()
    for i in range(len(args)):
        arg = args[i]

        if type(resolutions[parameter_names[i]]) == int:
            arg = int(round(arg))

        res += (arg,)

    return res


def get_suitable_objective_function_for_bayes_opt(
    objective_dict,
    strategy_type,
    resolutions,
    parameter_names,
    price_data,
    spread_data,
    balance,
    use_all_start_times=False,
    quantile=0.45,
    min_avg_monthly_return=1.10,
):
    if strategy_type == "ma":

        def objective_fn(aggregate_N, w):
            args = (aggregate_N, w)
            args = get_rounded_args(args, resolutions, parameter_names)

            if args not in objective_dict:
                objective_dict[args] = get_objective_function(
                    args,
                    strategy_type,
                    price_data,
                    spread_data,
                    balance,
                    use_all_start_times=use_all_start_times,
                    quantile=quantile,
                    min_avg_monthly_return=min_avg_monthly_return,
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
                    price_data,
                    spread_data,
                    balance,
                    use_all_start_times=use_all_start_times,
                    quantile=quantile,
                    min_avg_monthly_return=min_avg_monthly_return,
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
                    price_data,
                    spread_data,
                    balance,
                    use_all_start_times=use_all_start_times,
                    quantile=quantile,
                    min_avg_monthly_return=min_avg_monthly_return,
                )

            return objective_dict[args]["objective"]

    return objective_fn


def get_objective_function(
    args,
    strategy_type,
    price_data,
    spread_data,
    balance,
    use_all_start_times=False,
    quantile=0.45,
    min_avg_monthly_return=1.10,
):
    skip_cond, conflict_value = skip_condition(strategy_type, args)
    if skip_cond:
        return {"objective": conflict_value}

    get_buys_and_sells_fn = utils.choose_get_buys_and_sells_fn(strategy_type)

    sub_optim_results_dir = "optim_results/sub_optim_results/"
    if not os.path.exists(sub_optim_results_dir):
        os.mkdir(sub_optim_results_dir)

    joined_str_args = "_".join(str(arg) for arg in args)

    filename = sub_optim_results_dir + f"{strategy_type}_{joined_str_args}.json"

    if os.path.exists(filename):
        with open(filename, "r") as file:
            init_sub_args = json.load(file)

        init_points = 50
    else:
        init_sub_args = None
        init_points = 100

    optimized_params, objective, stats = optimize_sltp_and_balancing_params(
        init_sub_args,
        init_points,
        args,
        price_data,
        get_buys_and_sells_fn,
        spread_data,
        balance,
        use_all_start_times=use_all_start_times,
        quantile=quantile,
        min_avg_monthly_return=min_avg_monthly_return,
        debug=False,
    )

    with open(filename, "w") as file:
        json.dump(optimized_params, file)

    params_dict = {
        "params": args,
        "secondary_params": optimized_params,
        "objective": objective,
        "stats": stats,
    }

    return params_dict


def optimize_sltp_and_balancing_params(
    init_sub_args,
    init_points,
    args,
    price_data,
    get_buys_and_sells_fn,
    spread_data,
    balance,
    use_all_start_times=False,
    quantile=0.45,
    min_avg_monthly_return=1.05,
    debug=False,
):
    pbounds = {
        "stop_loss": (0.0, 0.99),
        "take_profit": (1.01, constants.MAX_TAKE_PROFIT),
        "target_usd_percentage": (0.0, 0.99),
        "balancing_period": (1, 7 * 24),
        "displacement": (0, 59),
    }

    stats = {}

    def sltp_and_balancing_objective_function(
        stop_loss, take_profit, target_usd_percentage, balancing_period, displacement
    ):
        balancing_period = int(round(balancing_period))
        displacement = int(round(displacement))

        (
            objective,
            median,
            mean,
            fraction_of_positives,
        ) = get_sltp_and_balancing_objective_function(
            args,
            price_data,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            displacement,
            get_buys_and_sells_fn,
            spread_data,
            balance,
            use_all_start_times=use_all_start_times,
            quantile=quantile,
            min_avg_monthly_return=min_avg_monthly_return,
        )

        stats[
            (
                stop_loss,
                take_profit,
                target_usd_percentage,
                balancing_period,
                displacement,
            )
        ] = {
            "median": median,
            "mean": mean,
            "fraction_of_positives": fraction_of_positives,
        }

        return objective

    optimizer = BayesianOptimization(
        f=sltp_and_balancing_objective_function,
        pbounds=pbounds,
        verbose=2 if debug else 0,
    )

    if init_sub_args is not None:
        optimizer.probe(params=init_sub_args, lazy=True)

    optimizer.maximize(
        init_points=init_points,
        n_iter=5,
    )

    # print(optimizer.space.params)
    # print(optimizer.space.target)
    # print(optimizer.res)
    # print()

    optimized_params = optimizer.max["params"]
    optimized_params["balancing_period"] = int(
        round(optimized_params["balancing_period"])
    )
    optimized_params["displacement"] = int(round(optimized_params["displacement"]))
    stats = stats[
        (
            optimized_params["stop_loss"],
            optimized_params["take_profit"],
            optimized_params["target_usd_percentage"],
            optimized_params["balancing_period"],
            optimized_params["displacement"],
        )
    ]

    objective = optimizer.max["target"]

    return optimized_params, objective, stats


# @profile
def get_sltp_and_balancing_objective_function(
    args,
    price_data,
    stop_loss,
    take_profit,
    target_usd_percentage,
    balancing_period,
    displacement,
    get_buys_and_sells_fn,
    spread_data,
    balance,
    use_all_start_times=False,
    quantile=0.45,
    min_avg_monthly_return=1.05,
    min_n_samples=10,
):
    aggregate_N, w = args[:2]

    monthly_log_profits_list = []

    # TODO: parallelize this?
    for symbol in price_data.keys():
        closes = price_data[symbol]["close"].values
        lows = price_data[symbol]["low"].values
        highs = price_data[symbol]["high"].values
        times = price_data[symbol]["time"].values

        first_time = price_data[symbol]["startTime"][0]
        first_timestamp = datetime.timestamp(parse_datetime(first_time))
        first_displacement = parse_datetime(first_time).minute

        displacements = utils.get_displacements(
            price_data[symbol]["time"].values, first_timestamp, first_displacement
        )

        tax_exemption = utils.get_leverage_from_symbol(symbol) == 1

        # TODO: cache these results?
        (
            aggregated_closes,
            aggregated_lows,
            aggregated_highs,
            split_indices,
            _,
        ) = utils.aggregate_price_data_from_displacement(
            closes, lows, highs, times, displacements, displacement
        )

        for i in range(len(split_indices) - 1):
            aggregated_closes_part = aggregated_closes[
                split_indices[i] : split_indices[i + 1]
            ]
            aggregated_lows_part = aggregated_lows[
                split_indices[i] : split_indices[i + 1]
            ]
            aggregated_highs_part = aggregated_highs[
                split_indices[i] : split_indices[i + 1]
            ]

            if len(aggregated_closes_part) > aggregate_N * w:
                buys, sells, N = get_buys_and_sells_fn(
                    aggregated_closes_part, *args, from_minute=False
                )
                aggregated_closes_part = aggregated_closes_part[-N:]
                aggregated_lows_part = aggregated_lows_part[-N:]
                aggregated_highs_part = aggregated_highs_part[-N:]

                if np.any(buys > 0.0):
                    trade_starts, trade_ends = get_trades(
                        buys, sells, aggregate_N, use_all_start_times
                    )
                    prev_remaining_log_profit = 0.0
                    prev_remaining_log_profits_len = 0

                    for trade_start, trade_end in zip(trade_starts, trade_ends):
                        trade_closes = (
                            aggregated_closes_part[trade_start + 1 : trade_end + 1]
                            / aggregated_closes_part[trade_start]
                        )
                        trade_lows = (
                            aggregated_lows_part[trade_start + 1 : trade_end + 1]
                            / aggregated_closes_part[trade_start]
                        )
                        trade_highs = (
                            aggregated_highs_part[trade_start + 1 : trade_end + 1]
                            / aggregated_closes_part[trade_start]
                        )

                        (
                            monthly_log_profits,
                            prev_remaining_log_profit,
                            prev_remaining_log_profits_len,
                        ) = sample_log_profits(
                            symbol,
                            trade_closes,
                            trade_lows,
                            trade_highs,
                            stop_loss,
                            take_profit,
                            target_usd_percentage,
                            balancing_period,
                            spread_data[symbol],
                            balance,
                            prev_remaining_log_profit,
                            prev_remaining_log_profits_len,
                            tax_exemption=tax_exemption,
                        )

                        monthly_log_profits_list.append(monthly_log_profits)

    if len(monthly_log_profits_list) == 0:
        return -1.0, 0.0, 0.0, 0.0

    monthly_log_profits = np.concatenate(monthly_log_profits_list)

    if len(monthly_log_profits) < min_n_samples:
        return -1.0, 0.0, 0.0, 0.0

    objective = np.exp(np.quantile(monthly_log_profits, quantile))
    median = np.exp(np.median(monthly_log_profits))
    mean = np.exp(np.mean(monthly_log_profits))

    n_positives = np.sum(monthly_log_profits > 0.0)
    n_negatives = np.sum(monthly_log_profits < 0.0)
    fraction_of_positives = n_positives / (n_positives + n_negatives)

    # probability of profitability should be at least 50% and average profit shoud
    # be positive (i.e. greater than 1.0)
    if median < 1.0 or mean <= min_avg_monthly_return:
        objective -= 1

    if np.isnan(objective):
        print(args)
        print(
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            displacement,
        )
        # import ipdb; ipdb.set_trace()

        raise ValueError("Objective should not be NaN.")

    return objective, median, mean, fraction_of_positives


def get_trades(buys, sells, aggregate_N, use_all_start_times):
    n = aggregate_N if use_all_start_times else 1

    trade_starts_list = []
    trade_ends_list = []

    for i in range(n):
        trade_starts, trade_ends = utils.get_entry_and_exit_idx(
            buys[i::aggregate_N], sells[i::aggregate_N]
        )

        trade_starts *= aggregate_N
        trade_ends *= aggregate_N

        trade_starts += i
        trade_ends += i

        trade_starts_list.append(trade_starts)
        trade_ends_list.append(trade_ends)

        # for trade_start, trade_end in zip(trade_starts, trade_ends):
        #     possible_trade_starts = np.arange(trade_start, trade_end, aggregate_N)
        #     trade_starts_list.append(possible_trade_starts)
        #     trade_ends_list.append(np.repeat(trade_end, len(possible_trade_starts)))

    if len(trade_starts_list) > 0:
        trade_starts = np.concatenate(trade_starts_list)
        trade_ends = np.concatenate(trade_ends_list)
    else:
        trade_starts = np.empty(0)
        trade_ends = np.empty(0)

    return trade_starts, trade_ends


@njit
def sample_log_profits(
    symbol,
    trade_closes,
    trade_lows,
    trade_highs,
    stop_loss,
    take_profit,
    target_usd_percentage,
    balancing_period,
    orderbook_distributions,
    balance,
    prev_remaining_log_profit,
    prev_remaining_log_profits_len,
    tax_exemption=False,
    eps=1e-7,
):

    stop_loss_index = get_stop_loss_index(trade_lows, stop_loss)
    take_profit_index = get_take_profit_index(trade_highs, take_profit)

    usd_values, asset_values, _ = get_balanced_trade(
        trade_closes,
        stop_loss_index,
        take_profit_index,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        orderbook_distributions,
        balance,
        taxes=True,
        tax_exemption=tax_exemption,
        from_minute=False,
    )

    trigger_i = min(stop_loss_index, take_profit_index)

    log_profits = np.log(usd_values + asset_values + eps)
    log_profits = np.concatenate((np.zeros(1), log_profits[: trigger_i + 1]))

    idx = np.arange(24 * 30 - prev_remaining_log_profits_len, len(log_profits), 24 * 30)

    if len(idx) > 0:
        idx = np.concatenate((np.zeros(1, dtype=np.int32), idx))
        monthly_log_profits = np.diff(log_profits[idx])
        monthly_log_profits[0] += prev_remaining_log_profit

        remaining_log_profits = log_profits[idx[-1] :]
        remaining_log_profit = 0.0
        remaining_log_profits_len = 0

    else:
        monthly_log_profits = np.zeros(0)

        remaining_log_profits = log_profits
        remaining_log_profit = prev_remaining_log_profit
        remaining_log_profits_len = prev_remaining_log_profits_len

    remaining_log_profit += remaining_log_profits[-1] - remaining_log_profits[0]
    remaining_log_profits_len += len(remaining_log_profits)

    return monthly_log_profits, remaining_log_profit, remaining_log_profits_len


def get_stop_loss_index_naive(lows, stop_loss: float):
    idx = np.argwhere(lows <= stop_loss)

    return idx[0].item() if len(idx) > 0 else len(lows)


@njit
def get_stop_loss_index(lows, stop_loss: float):
    for i in range(len(lows)):
        if lows[i] <= stop_loss:
            return i

    return len(lows)


def get_take_profit_index_naive(highs, take_profit: float):
    idx = np.argwhere(highs >= take_profit)

    return idx[0].item() if len(idx) > 0 else len(highs)


@njit
def get_take_profit_index(highs, take_profit: float):
    for i in range(len(highs)):
        if highs[i] >= take_profit:
            return i

    return len(highs)


# assumes trade is normalized to start from 1
def get_stop_loss_indices_naive(
    closes, lows, stop_loss: float, aggregate_N: int, from_minute=False
):
    assert len(closes) == len(lows)

    if from_minute:
        aggregate_N *= 60
    aggregate_N = min(aggregate_N, len(closes))

    stop_loss_indices = np.zeros(aggregate_N)

    for i in range(aggregate_N):
        # renormalize lows
        normalized_lows = lows[i:] / closes[i - 1] if i > 0 else lows
        stop_loss_indices[i] = get_stop_loss_index_naive(normalized_lows, stop_loss)

    return stop_loss_indices


# assumes trade is normalized to start from 1
@njit
def get_stop_loss_indices(
    closes, lows, stop_loss: float, aggregate_N: int, from_minute=False
):
    assert len(closes) == len(lows)

    if from_minute:
        aggregate_N *= 60
    aggregate_N = min(aggregate_N, len(closes))

    stop_loss_indices = np.zeros(aggregate_N)
    current_close = 1.0

    for i in range(aggregate_N):
        prev_close = current_close
        if i > 0:
            prev_stop_loss_index = stop_loss_indices[i - 1] + i - 1
            current_close = closes[i - 1]
        else:
            prev_stop_loss_index = 0
            current_close = 1.0

        # renormalize lows
        if current_close > prev_close and prev_stop_loss_index > i:
            # if current close is larger than previous close, then it suffices to check
            # only up to the previous stop loss index
            # exception: when previous stop loss was triggered before current close
            normalized_lows = lows[i:prev_stop_loss_index] / current_close
            stop_loss_indices[i] = get_stop_loss_index(normalized_lows, stop_loss)

        else:
            # if the current close is less than the previous close, then it suffices to
            # check only after the the previous stop loss index
            i_start = max(i, prev_stop_loss_index)
            normalized_lows = lows[i_start:] / current_close
            stop_loss_indices[i] = (
                get_stop_loss_index(normalized_lows, stop_loss) + i_start - i
            )

    return stop_loss_indices


# assumes trade chunk is normalized to start from 1
def get_take_profit_indices_naive(
    closes, highs, take_profit: float, aggregate_N: int, from_minute=False
):
    assert len(closes) == len(highs)

    if from_minute:
        aggregate_N *= 60
    aggregate_N = min(aggregate_N, len(closes))

    take_profit_indices = np.zeros(aggregate_N)

    for i in range(aggregate_N):
        # renormalize highs
        normalized_highs = highs[i:] / closes[i - 1] if i > 0 else highs
        take_profit_indices[i] = get_take_profit_index_naive(
            normalized_highs, take_profit
        )

    return take_profit_indices


# assumes trade chunk is normalized to start from 1
@njit
def get_take_profit_indices(
    closes, highs, take_profit: float, aggregate_N: int, from_minute=False
):
    assert len(closes) == len(highs)

    if from_minute:
        aggregate_N *= 60
    aggregate_N = min(aggregate_N, len(closes))

    take_profit_indices = np.zeros(aggregate_N)
    current_close = 1.0

    for i in range(aggregate_N):
        prev_close = current_close
        if i > 0:
            prev_take_profit_index = take_profit_indices[i - 1] + i - 1
            current_close = closes[i - 1]
        else:
            prev_take_profit_index = 0
            current_close = 1.0

        # renormalize highs
        if current_close < prev_close and prev_take_profit_index > i:
            # if current close is smaller than previous close, then it suffices to check
            # only up to the previous take profit index
            # exception: when previous take profit was triggered before current close
            normalized_highs = highs[i:prev_take_profit_index] / current_close
            take_profit_indices[i] = get_take_profit_index(
                normalized_highs, take_profit
            )

        else:
            # if the current close is more than the previous close, then it suffices to
            # check only after the the previous take profit index
            i_start = max(i, prev_take_profit_index)
            normalized_highs = highs[i_start:] / current_close
            take_profit_indices[i] = (
                get_take_profit_index(normalized_highs, take_profit) + i_start - i
            )

    return take_profit_indices


def get_balanced_trade_naive(
    closes,
    stop_loss_index,
    take_profit_index,
    stop_loss,
    take_profit,
    target_usd_percentage,
    balancing_period,
    orderbook_distributions,
    balance,
    taxes=True,
    tax_exemption=True,
    from_minute=False,
):
    N = len(closes)
    if from_minute:
        balancing_period *= 60

    usd_values = np.zeros(N + 1)
    usd_values[0] = 1.0
    asset_values = np.zeros(N + 1)

    trigger_i = min(stop_loss_index, take_profit_index)
    triggered = trigger_i < N
    if not triggered:
        trigger_i = N - 1

    N_balancings = trigger_i // balancing_period
    prices = np.ones(N_balancings + 2)
    buy_sizes = np.zeros(N_balancings + 2)
    buy_sizes_index = 0

    (
        usd_values[0],
        asset_values[0],
        new_buy_size,
        buy_size_diffs,
    ) = utils.get_balanced_usd_and_asset_values(
        usd_values[0],
        asset_values[0],
        target_usd_percentage,
        buy_prices=prices[:buy_sizes_index],
        buy_sizes=buy_sizes[:buy_sizes_index],
        price=prices[buy_sizes_index],
        balance=balance,
        orderbook_distributions=orderbook_distributions,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    buy_sizes[buy_sizes_index] = new_buy_size

    for i in range(balancing_period, trigger_i + 1, balancing_period):
        buy_sizes_index = i // balancing_period
        prices[buy_sizes_index] = closes[i - 1]

        norm_closes = (
            closes[i - balancing_period : i] / closes[i - balancing_period - 1]
            if i > balancing_period
            else closes[i - balancing_period : i]
        )
        asset_values[i - balancing_period + 1 : i + 1] = (
            asset_values[i - balancing_period] * norm_closes
        )

        usd_values[i - balancing_period + 1 : i + 1] = usd_values[i - balancing_period]

        (
            usd_values[i],
            asset_values[i],
            new_buy_size,
            buy_size_diffs,
        ) = utils.get_balanced_usd_and_asset_values(
            usd_values[i],
            asset_values[i],
            target_usd_percentage,
            buy_prices=prices[:buy_sizes_index],
            buy_sizes=buy_sizes[:buy_sizes_index],
            price=prices[buy_sizes_index],
            balance=balance,
            orderbook_distributions=orderbook_distributions,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        buy_sizes[buy_sizes_index] = new_buy_size
        buy_sizes[:buy_sizes_index] -= buy_size_diffs

    if trigger_i // balancing_period > 0:
        assert i == (trigger_i // balancing_period) * balancing_period
        norm_closes = closes[i : trigger_i + 1] / closes[i - 1]
    else:
        i = 0
        norm_closes = closes[i : trigger_i + 1]

    asset_values[i + 1 : trigger_i + 2] = asset_values[i] * norm_closes
    usd_values[i + 1 : trigger_i + 2] = usd_values[i]

    usd_values = usd_values[1:]
    asset_values = asset_values[1:]

    stop_loss_is_before_take_profit = stop_loss_index <= take_profit_index

    assert prices[-1] == 1.0
    buy_sizes_index = len(buy_sizes) - 1

    if triggered:
        trigger_value = stop_loss if stop_loss_is_before_take_profit else take_profit
        asset_values[trigger_i] *= trigger_value / closes[trigger_i]

        prices[buy_sizes_index] = trigger_value
    else:
        prices[buy_sizes_index] = closes[-1]

    (
        usd_values[trigger_i],
        asset_values[trigger_i],
        new_buy_size,
        buy_size_diffs,
    ) = utils.get_balanced_usd_and_asset_values(
        usd_values[trigger_i],
        asset_values[trigger_i],
        1.0,
        buy_prices=prices[:buy_sizes_index],
        buy_sizes=buy_sizes[:buy_sizes_index],
        price=prices[buy_sizes_index],
        balance=balance,
        orderbook_distributions=orderbook_distributions,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    buy_sizes[buy_sizes_index] = new_buy_size
    buy_sizes[:buy_sizes_index] -= buy_size_diffs

    usd_values[trigger_i:] = usd_values[trigger_i]
    asset_values[trigger_i:] = asset_values[trigger_i]

    return usd_values, asset_values, buy_sizes


get_balanced_trade = njit()(get_balanced_trade_naive)


def combine_trade_chunks():
    pass


def calculate_corr_matrix(data_dir, coins):
    timeseries = []

    for coin in coins:
        data_file = os.path.join(data_dir, coin, coin + ".csv")
        if os.path.exists(data_file):
            symbol = coin
        else:
            symbol = utils.get_symbol(coin, m=3)

        symbol_price_data = data.load_price_data(
            data_dir,
            symbol + "/USD",
            return_price_data=True,
            return_price_data_only=True,
        )

        # TODO: rename close
        timeseries.append(symbol_price_data[["startTime", "close"]])

        print(len(symbol_price_data))

    combined_series = combine_time_series_by_common_time(timeseries)
    common_time = combined_series["startTime"]

    print(combined_series)


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


def skip_condition(strategy_type, args):
    if strategy_type == "macross":
        _, w, w2 = args

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


def get_balanced_wealths(
    X,
    weight_values,
    keys,
    potential_spreads,
    potential_balances,
    total_balance,
    balancing_period=1,
    taxes=False,
):
    idx = np.arange(0, X.shape[0], balancing_period)
    X = X[idx, :]

    X_change = X[1:, :] / X[:-1, :]

    new_weights = X_change * weight_values.reshape(1, -1)
    balanced_wealths = np.zeros((len(X),))

    for i in range(new_weights.shape[0]):
        new_weights_relative = new_weights[i] / np.sum(new_weights[i])
        if taxes:
            new_weights_relative = (
                apply_taxes(new_weights_relative / weight_values) * weight_values
            )
            new_weights[i] = new_weights_relative * np.sum(new_weights[i])

        li_neg = X_change[i, :] < 1
        li_pos = ~li_neg

        total_commisions_and_spread = []

        spread_dict = {}

        for j, key in enumerate(keys):
            coin = key.split("_")[0]
            m, m_bear = tuple([int(x) for x in key.split("_")[3:]])

            if coin not in spread_dict:
                spread_dict[coin] = potential_spreads[(coin, m, m_bear)][
                    np.argmin(
                        np.abs(
                            potential_balances
                            - total_balance
                            * new_weights_relative[j]
                            * balanced_wealths[i]
                        )
                    )
                ]

            spread = spread_dict[coin]

            total_commisions_and_spread.append(
                np.abs(weight_values[j] - new_weights_relative[j])
                * (commissions + spread)
            )

        total_commisions_and_spread = np.array(total_commisions_and_spread)
        total_commisions_and_spread = np.log(
            1 - np.sum(total_commisions_and_spread[li_neg])
        ) + np.log(1 - np.sum(total_commisions_and_spread[li_pos]))
        balanced_wealths[i + 1] = (
            balanced_wealths[i]
            + np.log(np.sum(new_weights[i]))
            + total_commisions_and_spread
        )

    balanced_wealths = np.exp(balanced_wealths)

    return balanced_wealths


def return_trade_wealths(
    trade_wealths, total_months, taxes=True, return_total=True, return_as_log=True
):
    if taxes:
        trade_wealths = apply_taxes(trade_wealths)

    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return res if return_as_log else np.exp(res)

    return np.log(trade_wealths) if return_as_log else trade_wealths


# TODO: add comments or split this into multiple functions
def get_adaptive_wealths(
    X,
    buys,
    sells,
    aggregate_N,
    sltp_args,
    total_balance,
    potential_balances,
    potential_spreads,
    place_take_profit_and_stop_loss_simultaneously=False,
    trail_value_recalc_period=None,
    commissions=0.0007,
    short=False,
    X_bear=None,
    taxes=True,
):

    # TODO: move this into a separate function?
    spread = potential_spreads[np.argmin(np.abs(potential_balances - total_balance))]
    commissions_and_spread = 1 - commissions - spread

    # assert(short == (len(sltp_args) == 4))
    assert short == (X_bear is not None)

    N = buys.shape[0]
    buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)

    if trail_value_recalc_period is None:
        trail_value_recalc_period = N + 1

    wealths = np.ones((N,))

    if buys_idx[0] < sells_idx[0]:
        buy_state = True
        entry_i = buys_idx[0]
    else:
        buy_state = False
        entry_i = sells_idx[0]

    entry_is_transaction = buy_state or short

    # handle place_take_profit_and_stop_loss_simultaneously = True
    if place_take_profit_and_stop_loss_simultaneously:
        trigger_names_and_params = {}
        trigger_names_and_params[True] = sltp_args[1:4]
        if short:
            trigger_names_and_params[False] = sltp_args[5:]

        trigger_triggered = False

        while entry_i < N - 1:
            exit_is_transaction = True
            if buy_state:
                larger_exits_idx = sells_idx[sells_idx > entry_i]
            else:
                larger_exits_idx = buys_idx[buys_idx > entry_i]
            exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
            if (exit_i - 1 - entry_i > aggregate_N * 60) and (buy_state or short):
                exit_i = entry_i + aggregate_N * 60 + 1
                exit_is_transaction = False

            if buy_state:
                price = X[entry_i, 0]
                sub_X = X[entry_i:exit_i, :] / X[entry_i, 0]
            elif short:
                price = X_bear[entry_i, 0]
                sub_X = X_bear[entry_i:exit_i, :] / X_bear[entry_i, 0]

            old_wealth = wealths[entry_i]

            if not (buy_state or short):
                wealths[entry_i:exit_i] = old_wealth
            else:
                if entry_is_transaction:
                    old_wealth *= commissions_and_spread

                    buy_price = price
                    buy_i = entry_i

                    take_profit, stop_loss_type, stop_loss = trigger_names_and_params[
                        buy_state
                    ]
                    take_profit_price = buy_price * take_profit
                    stop_loss_price = buy_price * stop_loss

                    trigger_triggered = False
                    max_price = price

                    trail_value_recalc_idx = np.arange(
                        buy_i + 1, N, trail_value_recalc_period
                    )

                if not trigger_triggered:
                    take_profit_prices = np.repeat(
                        take_profit_price, exit_i - entry_i - 1
                    )

                    if stop_loss_type != "trailing":
                        stop_loss_prices1 = stop_loss_prices2 = np.repeat(
                            stop_loss_price, exit_i - entry_i - 1
                        )
                    else:
                        li = (trail_value_recalc_idx > entry_i + 1) & (
                            trail_value_recalc_idx < exit_i
                        )
                        starts_and_ends = trail_value_recalc_idx[li]
                        starts_and_ends = np.concatenate(
                            [
                                np.array([entry_i + 1]),
                                starts_and_ends,
                                np.array([exit_i]),
                            ]
                        )

                        stop_loss_prices1 = np.zeros((exit_i - entry_i - 1,))
                        stop_loss_prices2 = np.zeros((exit_i - entry_i - 1,))

                        for start, end in zip(
                            starts_and_ends[:-1], starts_and_ends[1:]
                        ):
                            if (start - buy_i - 1) % trail_value_recalc_period == 0:
                                acc_max_start_price = (
                                    sub_X[start - entry_i - 1, 0] * price
                                )

                                if start - buy_i - 1 == 0:
                                    trail_value = (stop_loss - 1.0) * buy_price
                                else:
                                    trail_value = (
                                        stop_loss_prices2[start - entry_i - 2]
                                        - sub_X[start - entry_i - 1, 0] * price
                                    )
                            else:
                                acc_max_start_price = max_price

                            accumulative_max = np.maximum.accumulate(
                                np.concatenate(
                                    [
                                        np.array([acc_max_start_price]),
                                        sub_X[(start - entry_i) : (end - entry_i), 1]
                                        * price,
                                    ]
                                )
                            )
                            max_price = accumulative_max[-1]
                            # for each candle/timestep, low occurs before high
                            stop_loss_prices1[
                                (start - entry_i - 1) : (end - entry_i - 1)
                            ] = (accumulative_max[:-1] + trail_value)
                            # for each candle/timestep, high occurs before low
                            stop_loss_prices2[
                                (start - entry_i - 1) : (end - entry_i - 1)
                            ] = (accumulative_max[1:] + trail_value)

                        assert np.all(np.diff(stop_loss_prices1) >= 0)
                        assert np.all(np.diff(stop_loss_prices2) >= 0)

                    li_take_profit = sub_X[1:, 1] * price > take_profit_prices
                    cond_take_profit = np.any(li_take_profit)

                    li_stop_loss1 = sub_X[1:, 2] * price < stop_loss_prices1
                    if stop_loss_type == "trailing":
                        li_stop_loss2 = sub_X[1:, 2] * price < stop_loss_prices2
                    else:
                        li_stop_loss2 = li_stop_loss1

                    cond_stop_loss1 = np.any(li_stop_loss1)
                    cond_stop_loss2 = np.any(li_stop_loss2)

                # check if trigger is triggered
                if not trigger_triggered and (
                    cond_stop_loss1 or cond_stop_loss2 or cond_take_profit
                ):
                    idx = np.arange(len(sub_X) - 1) + 1

                    if cond_take_profit:
                        i_take_profit = idx[li_take_profit][0]

                    if cond_stop_loss1:
                        idx1 = idx[li_stop_loss1]
                        i_stop_loss1 = idx1[0]
                        stop_loss_price1 = stop_loss_prices1[li_stop_loss1][0]
                        trigger_triggered = True
                    else:
                        i_stop_loss1 = len(sub_X) - 1
                        stop_loss_price1 = sub_X[-1, 0] * price

                    if stop_loss_type != "trailing":
                        i_stop_loss = i_stop_loss1
                        stop_loss_price = stop_loss_price1

                    if stop_loss_type == "trailing" and cond_stop_loss2:
                        idx2 = idx[li_stop_loss2]
                        i_stop_loss2 = idx2[0]

                        stop_loss_price2 = (
                            max(
                                stop_loss_prices1[li_stop_loss2][0],
                                sub_X[i_stop_loss2, 2] * price,
                            )
                            + stop_loss_prices2[li_stop_loss2][0]
                        ) / 2

                        possible_trigger_prices = np.array(
                            [stop_loss_price1, stop_loss_price2]
                        )
                        possible_trigger_prices_min_i = np.argmin(
                            possible_trigger_prices
                        )
                        stop_loss_price = possible_trigger_prices[
                            possible_trigger_prices_min_i
                        ]

                        i_stop_losses = np.array([i_stop_loss1, i_stop_loss2])
                        i_stop_loss = i_stop_losses[possible_trigger_prices_min_i]

                        if i_stop_loss == i_stop_loss2:
                            trigger_triggered = True

                    if cond_stop_loss2:
                        if cond_take_profit:
                            if stop_loss_type == "trailing":
                                # handle both cases of triggered trailing prices here (not just the minimum)
                                # because the max triggered trailing price is still smaller than take profit
                                if i_stop_loss <= i_take_profit:
                                    trigger_price = stop_loss_price
                                    min_i = i_stop_loss
                                elif (
                                    i_stop_losses[1 - possible_trigger_prices_min_i]
                                    <= i_take_profit
                                ):
                                    trigger_price = possible_trigger_prices[
                                        1 - possible_trigger_prices_min_i
                                    ]
                                    min_i = i_stop_losses[
                                        1 - possible_trigger_prices_min_i
                                    ]
                                else:
                                    trigger_price = take_profit_price
                                    trigger_triggered = True
                                    min_i = i_take_profit
                            else:
                                # cond_stop_loss1 is equal to cond_stop_loss2
                                if i_stop_loss <= i_take_profit:
                                    trigger_price = stop_loss_price
                                    min_i = i_stop_loss
                                else:
                                    trigger_price = take_profit_price
                                    trigger_triggered = True
                                    min_i = i_take_profit
                        else:
                            trigger_price = stop_loss_price
                            min_i = i_stop_loss
                    elif cond_take_profit:
                        trigger_price = take_profit_price
                        trigger_triggered = True
                        min_i = i_take_profit

                    wealths[entry_i : entry_i + min_i] = sub_X[:min_i, 0] * old_wealth

                    new_wealth = trigger_price / price * old_wealth
                    if trigger_price > buy_price and taxes:
                        profit = trigger_price / buy_price
                        new_wealth *= apply_taxes(profit, copy=True) / profit

                    wealths[entry_i + min_i : exit_i] = new_wealth

                    if trigger_triggered:
                        spread = potential_spreads[
                            np.argmin(
                                np.abs(potential_balances - new_wealth * total_balance)
                            )
                        ]
                        commissions_and_spread = 1 - commissions - spread

                        wealths[entry_i + min_i : exit_i] *= commissions_and_spread

                elif trigger_triggered:
                    wealths[entry_i:exit_i] = old_wealth
                else:
                    wealths[entry_i:exit_i] = sub_X[:, 0] * old_wealth

            if (
                exit_is_transaction
                and not trigger_triggered
                and (buy_state or short)
                and taxes
            ):
                sell_price = sub_X[-1, 0] * price
                if sell_price > buy_price:
                    profit = sell_price / buy_price
                    wealths[exit_i - 1] *= apply_taxes(profit, copy=True) / profit

            spread = potential_spreads[
                np.argmin(
                    np.abs(potential_balances - wealths[exit_i - 1] * total_balance)
                )
            ]
            commissions_and_spread = 1 - commissions - spread

            if exit_is_transaction and not trigger_triggered and (buy_state or short):
                wealths[exit_i - 1] *= commissions_and_spread

            entry_i = exit_i - 1
            if exit_is_transaction:
                buy_state = not buy_state
                entry_is_transaction = True
            else:
                entry_is_transaction = False

    else:
        trigger_names_and_params = {}
        trigger_names_and_params[True] = sltp_args[:2]
        if short:
            trigger_names_and_params[False] = sltp_args[2:]

        trigger_triggered = False

        while entry_i < N - 1:
            exit_is_transaction = True
            if buy_state:
                larger_exits_idx = sells_idx[sells_idx > entry_i]
            else:
                larger_exits_idx = buys_idx[buys_idx > entry_i]
            exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
            if (exit_i - 1 - entry_i > aggregate_N * 60) and (buy_state or short):
                exit_i = entry_i + aggregate_N * 60 + 1
                exit_is_transaction = False

            if buy_state:
                price = X[entry_i, 0]
                sub_X = X[entry_i:exit_i, :] / X[entry_i, 0]
            elif short:
                price = X_bear[entry_i, 0]
                sub_X = X_bear[entry_i:exit_i, :] / X_bear[entry_i, 0]

            old_wealth = wealths[entry_i]

            if not (buy_state or short):
                wealths[entry_i:exit_i] = old_wealth
            else:
                if entry_is_transaction:
                    old_wealth *= commissions_and_spread

                    buy_price = price
                    buy_i = entry_i

                    trigger_name, trigger_param = trigger_names_and_params[buy_state]
                    trigger_price = buy_price * trigger_param
                    trigger_triggered = False
                    max_price = price

                    trail_value_recalc_idx = np.arange(
                        buy_i + 1, N, trail_value_recalc_period
                    )

                if not trigger_triggered:
                    if trigger_name != "trailing":
                        trigger_prices1 = trigger_prices2 = np.repeat(
                            trigger_price, exit_i - entry_i - 1
                        )
                    else:
                        # TODO: adapt to the real world scenario where trail_value is updated
                        # (s.t. trigger price won't decrease), and accumulative_max is reset

                        li = (trail_value_recalc_idx > entry_i + 1) & (
                            trail_value_recalc_idx < exit_i
                        )
                        starts_and_ends = trail_value_recalc_idx[li]
                        starts_and_ends = np.concatenate(
                            [
                                np.array([entry_i + 1]),
                                starts_and_ends,
                                np.array([exit_i]),
                            ]
                        )

                        trigger_prices1 = np.zeros((exit_i - entry_i - 1,))
                        trigger_prices2 = np.zeros((exit_i - entry_i - 1,))

                        for start, end in zip(
                            starts_and_ends[:-1], starts_and_ends[1:]
                        ):
                            if (start - buy_i - 1) % trail_value_recalc_period == 0:
                                acc_max_start_price = (
                                    sub_X[start - entry_i - 1, 0] * price
                                )

                                if start - buy_i - 1 == 0:
                                    trail_value = (trigger_param - 1.0) * buy_price
                                else:
                                    trail_value = (
                                        trigger_prices2[start - entry_i - 2]
                                        - sub_X[start - entry_i - 1, 0] * price
                                    )
                            else:
                                acc_max_start_price = max_price

                            accumulative_max = np.maximum.accumulate(
                                np.concatenate(
                                    [
                                        np.array([acc_max_start_price]),
                                        sub_X[(start - entry_i) : (end - entry_i), 1]
                                        * price,
                                    ]
                                )
                            )
                            max_price = accumulative_max[-1]
                            # for each candle/timestep, low occurs before high
                            trigger_prices1[
                                (start - entry_i - 1) : (end - entry_i - 1)
                            ] = (accumulative_max[:-1] + trail_value)
                            # for each candle/timestep, high occurs before low
                            trigger_prices2[
                                (start - entry_i - 1) : (end - entry_i - 1)
                            ] = (accumulative_max[1:] + trail_value)

                        assert np.all(np.diff(trigger_prices1) >= 0)
                        assert np.all(np.diff(trigger_prices2) >= 0)

                    if trigger_name == "take_profit":
                        li1 = li2 = sub_X[1:, 1] * price > trigger_prices1
                    else:
                        li1 = sub_X[1:, 2] * price < trigger_prices1
                        if trigger_name == "trailing":
                            li2 = sub_X[1:, 2] * price < trigger_prices2
                        else:
                            li2 = li1

                    cond1 = np.any(li1)
                    cond2 = np.any(li2)

                # check if trigger is triggered
                if not trigger_triggered and (cond1 or cond2):
                    idx = np.arange(len(sub_X) - 1) + 1

                    if cond1:
                        idx1 = idx[li1]
                        min_i = idx1[0]
                        trigger_price = trigger_prices1[li1][0]
                        trigger_triggered = True
                    else:
                        min_i = len(sub_X) - 1
                        trigger_price = sub_X[-1, 0] * price

                    if trigger_name == "trailing":
                        idx2 = idx[li2]
                        min_i2 = idx2[0]

                        prev_trigger_price = trigger_price
                        trigger_price = np.min(
                            [
                                trigger_price,
                                (
                                    max(
                                        trigger_prices1[li2][0],
                                        sub_X[min_i2, 2] * price,
                                    )
                                    + trigger_prices2[li2][0]
                                )
                                / 2,
                            ]
                        )

                        if trigger_price != prev_trigger_price:
                            trigger_triggered = True
                            min_i = min_i2

                    wealths[entry_i : entry_i + min_i] = sub_X[:min_i, 0] * old_wealth

                    new_wealth = trigger_price / price * old_wealth
                    if trigger_price > buy_price and taxes:
                        profit = trigger_price / buy_price
                        new_wealth *= apply_taxes(profit, copy=True) / profit

                    wealths[entry_i + min_i : exit_i] = new_wealth

                    if trigger_triggered:
                        spread = potential_spreads[
                            np.argmin(
                                np.abs(potential_balances - new_wealth * total_balance)
                            )
                        ]
                        commissions_and_spread = 1 - commissions - spread

                        wealths[entry_i + min_i : exit_i] *= commissions_and_spread

                elif trigger_triggered:
                    wealths[entry_i:exit_i] = old_wealth
                else:
                    wealths[entry_i:exit_i] = sub_X[:, 0] * old_wealth

            if (
                exit_is_transaction
                and not trigger_triggered
                and (buy_state or short)
                and taxes
            ):
                sell_price = sub_X[-1, 0] * price
                if sell_price > buy_price:
                    profit = sell_price / buy_price
                    wealths[exit_i - 1] *= apply_taxes(profit, copy=True) / profit

            spread = potential_spreads[
                np.argmin(
                    np.abs(potential_balances - wealths[exit_i - 1] * total_balance)
                )
            ]
            commissions_and_spread = 1 - commissions - spread

            if exit_is_transaction and not trigger_triggered and (buy_state or short):
                wealths[exit_i - 1] *= commissions_and_spread

            entry_i = exit_i - 1
            if exit_is_transaction:
                buy_state = not buy_state
                entry_is_transaction = True
            else:
                entry_is_transaction = False

    return wealths


def apply_commissions_and_spreads(
    wealth, N_transactions, commissions, spread, n_months=1, from_log=False
):
    commissions_and_spread = 1 - commissions - spread

    if from_log:
        res = wealth + np.log(commissions_and_spread) * N_transactions
        return np.exp(res / n_months)

    res = wealth * commissions_and_spread ** N_transactions
    return np.exp(np.log(res) / n_months)


# TODO: rename "trade_wealths"
# No commissions or spread
def get_trade_wealths(X, entries, exits, N):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    trade_wealths = []
    max_trade_wealths = []
    min_trade_wealths = []

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_X = X[entry_i:exit_i, :]
        if sub_X.shape[0] > 1:
            trade_wealths.append(sub_X[-1, 0] / sub_X[0, 0])
            max_trade_wealths.append(np.max(sub_X[1:, 1]) / sub_X[0, 0])  # use highs
            min_trade_wealths.append(np.min(sub_X[1:, 2]) / sub_X[0, 0])  # use lows

    return trade_wealths, min_trade_wealths, max_trade_wealths


# No commissions or spread
def get_sub_trade_wealths(X, entries, exits, N):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    sub_trade_wealths = []
    max_sub_trade_wealths = []
    min_sub_trade_wealths = []

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_X = X[entry_i:exit_i, :]
        sub_wealths = sub_X[1:, 0] / sub_X[0, 0]
        max_sub_wealths = sub_X[1:, 1] / sub_X[0, 0]
        min_sub_wealths = sub_X[1:, 2] / sub_X[0, 0]
        if sub_wealths.size > 0:
            sub_trade_wealths.append(sub_wealths)
            max_sub_trade_wealths.append(max_sub_wealths)
            min_sub_trade_wealths.append(min_sub_wealths)

    return sub_trade_wealths, min_sub_trade_wealths, max_sub_trade_wealths


# No commissions or spread
def get_take_profit_wealths_from_trades(
    trade_wealths,
    max_trade_wealths,
    take_profit,
    total_months,
    return_total=True,
    return_as_log=False,
    taxes=True,
):
    trade_wealths = np.array(trade_wealths)
    max_trade_wealths = np.array(max_trade_wealths)
    trade_wealths[max_trade_wealths > take_profit] = take_profit

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes=taxes,
        return_total=return_total,
        return_as_log=return_as_log,
    )


# No commissions or spread
def get_stop_loss_wealths_from_trades(
    trade_wealths,
    min_trade_wealths,
    stop_loss,
    total_months,
    return_total=True,
    return_as_log=False,
    taxes=True,
):
    trade_wealths = np.array(trade_wealths)
    min_trade_wealths = np.array(min_trade_wealths)
    trade_wealths[min_trade_wealths < stop_loss] = stop_loss

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes=taxes,
        return_total=return_total,
        return_as_log=return_as_log,
    )


# No commissions or spread
def get_trailing_stop_loss_wealths_from_sub_trades(
    sub_trade_wealths,
    min_sub_trade_wealths,
    max_sub_trade_wealths,
    trailing_stop_loss,
    total_months,
    trail_value_recalc_period=None,
    return_total=True,
    return_as_log=False,
    taxes=True,
):
    trade_wealths = np.array(
        [sub_trade_wealths[i][-1] for i in range(len(sub_trade_wealths))]
    )

    if trail_value_recalc_period is None:
        trail_value_recalc_period = np.max([len(x) for x in sub_trade_wealths]) + 1

    for i in range(len(sub_trade_wealths)):
        n = len(sub_trade_wealths[i])
        starts = np.arange(0, n, trail_value_recalc_period)
        ends = np.concatenate([starts[1:], np.array([n])])
        starts_and_ends = np.stack([starts, ends], axis=-1)

        trigger_prices1 = np.zeros((n,))
        trigger_prices2 = np.zeros((n,))

        for start, end in starts_and_ends:
            # update trail_value like in the real world (trigger price won't decrease)
            if start == 0:
                trail_value = trailing_stop_loss - 1.0
            else:
                trail_value = (
                    trigger_prices2[start - 1] - sub_trade_wealths[i][start - 1]
                )

            # accumulative_max is reset
            accumulative_max = np.maximum.accumulate(
                np.concatenate(
                    [
                        np.array(
                            [1.0 if start == 0 else sub_trade_wealths[i][start - 1]]
                        ),
                        max_sub_trade_wealths[i][start:end],
                    ]
                )
            )

            # for each candle/timestep, low occurs before high
            trigger_prices1[start:end] = accumulative_max[:-1] + trail_value
            # for each candle/timestep, high occurs before low
            trigger_prices2[start:end] = accumulative_max[1:] + trail_value

        assert np.all(np.diff(trigger_prices1) >= 0)
        assert np.all(np.diff(trigger_prices2) >= 0)

        li1 = min_sub_trade_wealths[i] < trigger_prices1
        li2 = min_sub_trade_wealths[i] < trigger_prices2

        cond1 = np.any(li1)
        cond2 = np.any(li2)

        # not possible to have cond1 == True and cond2 == False
        assert not (cond1 == True and cond2 == False)

        if cond2:
            # assume the worst case
            trade_wealths[i] = np.min(
                [
                    trigger_prices1[li1][0] if cond1 else trade_wealths[i],
                    (
                        max(trigger_prices1[li2][0], min_sub_trade_wealths[i][li2][0])
                        + trigger_prices2[li2][0]
                    )
                    / 2,
                ]
            )

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes=taxes,
        return_total=return_total,
        return_as_log=return_as_log,
    )


# No commissions or spread
def get_stop_loss_take_profit_wealths_from_sub_trades(
    sub_trade_wealths,
    min_sub_trade_wealths,
    max_sub_trade_wealths,
    stop_loss,
    stop_loss_type_is_trailing,
    take_profit,
    total_months=1,
    trail_value_recalc_period=None,
    return_total=True,
    return_as_log=False,
    return_latest_trigger_status=False,
    taxes=True,
):
    trade_wealths = np.array(
        [sub_trade_wealths[i][-1] for i in range(len(sub_trade_wealths))]
    )

    if trail_value_recalc_period is None:
        trail_value_recalc_period = np.max([len(x) for x in sub_trade_wealths]) + 1

    for i in range(len(sub_trade_wealths)):
        n = len(sub_trade_wealths[i])
        idx = np.arange(n)

        li_take_profit = max_sub_trade_wealths[i] > take_profit
        cond_take_profit = np.any(li_take_profit)

        if cond_take_profit:
            i_take_profit = idx[li_take_profit][0]

        if stop_loss_type_is_trailing:
            starts = np.arange(0, n, trail_value_recalc_period)
            ends = np.concatenate([starts[1:], np.array([n])])
            starts_and_ends = np.stack([starts, ends], axis=-1)

            trigger_prices1 = np.zeros((n,))
            trigger_prices2 = np.zeros((n,))

            for start, end in starts_and_ends:
                # update trail_value like in the real world (trigger price won't decrease)
                if start == 0:
                    trail_value = stop_loss - 1.0
                else:
                    trail_value = (
                        trigger_prices2[start - 1] - sub_trade_wealths[i][start - 1]
                    )

                # accumulative_max is reset
                accumulative_max = np.maximum.accumulate(
                    np.concatenate(
                        [
                            np.array(
                                [1.0 if start == 0 else sub_trade_wealths[i][start - 1]]
                            ),
                            max_sub_trade_wealths[i][start:end],
                        ]
                    )
                )

                # for each candle/timestep, low occurs before high
                trigger_prices1[start:end] = accumulative_max[:-1] + trail_value
                # for each candle/timestep, high occurs before low
                trigger_prices2[start:end] = accumulative_max[1:] + trail_value

            assert np.all(np.diff(trigger_prices1) >= 0)
            assert np.all(np.diff(trigger_prices2) >= 0)

            li_stop_loss1 = min_sub_trade_wealths[i] < trigger_prices1
            li_stop_loss2 = min_sub_trade_wealths[i] < trigger_prices2

            cond_stop_loss1 = np.any(li_stop_loss1)
            cond_stop_loss2 = np.any(li_stop_loss2)

            # not possible to have cond_stop_loss1 == True and cond_stop_loss2 == False
            assert not (cond_stop_loss1 == True and cond_stop_loss2 == False)

            if cond_stop_loss2:
                i_stop_loss1 = idx[li_stop_loss1][0] if cond_stop_loss1 else n + 1
                i_stop_loss2 = idx[li_stop_loss2][0]
                idx_stop_loss = np.array([i_stop_loss1, i_stop_loss2])

                # assume the worst case
                possible_trigger_prices = np.array(
                    [
                        trigger_prices1[i_stop_loss1]
                        if cond_stop_loss1
                        else trade_wealths[i],
                        (
                            max(
                                trigger_prices1[i_stop_loss2],
                                min_sub_trade_wealths[i][i_stop_loss2],
                            )
                            + trigger_prices2[i_stop_loss2]
                        )
                        / 2,
                    ]
                )

                possible_trigger_prices_min_i = np.argmin(possible_trigger_prices)

                stop_loss_trigger_price = possible_trigger_prices[
                    possible_trigger_prices_min_i
                ]
                i_stop_loss = idx_stop_loss[possible_trigger_prices_min_i]
        else:
            li_stop_loss = min_sub_trade_wealths[i] < stop_loss
            cond_stop_loss2 = np.any(li_stop_loss)
            stop_loss_trigger_price = stop_loss
            if cond_stop_loss2:
                i_stop_loss = idx[li_stop_loss][0]

        trigger_triggered = False

        if cond_stop_loss2:
            trigger_triggered = True
            if cond_take_profit:
                if stop_loss_type_is_trailing:
                    # handle both cases of triggered trailing prices here (not just the minimum)
                    # because the max triggered trailing price is still smaller than take profit
                    if i_stop_loss <= i_take_profit:
                        trade_wealths[i] = stop_loss_trigger_price
                    elif (
                        idx_stop_loss[1 - possible_trigger_prices_min_i]
                        <= i_take_profit
                    ):
                        trade_wealths[i] = possible_trigger_prices[
                            1 - possible_trigger_prices_min_i
                        ]
                    else:
                        trade_wealths[i] = take_profit
                else:
                    if i_stop_loss <= i_take_profit:
                        trade_wealths[i] = stop_loss_trigger_price
                    else:
                        trade_wealths[i] = take_profit
            else:
                trade_wealths[i] = stop_loss_trigger_price
        elif cond_take_profit:
            trigger_triggered = True
            trade_wealths[i] = take_profit

    if return_latest_trigger_status:
        return (
            return_trade_wealths(
                trade_wealths,
                total_months,
                taxes=taxes,
                return_total=return_total,
                return_as_log=return_as_log,
            ),
            trigger_triggered,
        )

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes=taxes,
        return_total=return_total,
        return_as_log=return_as_log,
    )


# TODO: make a single function for calculating wealths
#   - both as separate trades and a full time series
#       - option to combine separate trades into one
#   - needs to handle spreads adaptively
#   - how to handle generating inputs for portfolio optimization (each strategy
#     performance needs to be the same length with exactly the same time stamps)?
#   - how to handle short side?


def combine_sub_trade_wealths(sub_trade_wealths_long, sub_trade_wealths_short=None):
    pass


if __name__ == "__main__":
    pass
