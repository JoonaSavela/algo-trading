import os
import sys

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import algtra.optimize.utils as opt
from algtra.collect import data
from algtra import utils, constants
import numpy as np
import pandas as pd
import glob
import json
import time
from tqdm import tqdm
from itertools import product
import keys
from ftx.rest.client import FtxClient
import matplotlib.pyplot as plt
from datetime import datetime
from ciso8601 import parse_datetime


def get_moving_volumes(markets, displacement, k=None, return_first_timestamps=False):
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    if k is not None:
        markets = markets[:k]

    volumes_list = []
    first_timestamps = {}

    for market in tqdm(markets):
        price_data = data.load_price_data(data_dir, market, return_price_data_only=True)

        if price_data is not None:
            symbol = market.split("/")[0]

            times = price_data["time"].values
            volumes = price_data["volume"].values

            first_time = price_data["startTime"][0]
            first_timestamp = datetime.timestamp(parse_datetime(first_time))
            first_timestamps[symbol] = first_timestamp
            first_displacement = parse_datetime(first_time).minute

            displacements = utils.get_displacements(
                price_data["time"].values, first_timestamp, first_displacement
            )

            (
                aggregated_volumes,
                aggregated_times,
                _,
                _,
            ) = utils.aggregate_volume_data_from_displacement(
                volumes, times, displacements, displacement
            )

            if len(aggregated_volumes) >= 24:
                moving_volumes, times = utils.gap_aware_moving_sum(
                    aggregated_volumes, aggregated_times, window_size=24
                )

                volumes_list.append(
                    pd.DataFrame(
                        {
                            symbol: moving_volumes,
                            "time": times,
                        }
                    )
                )

    moving_volumes = utils.combine_time_series_by_common_time(
        volumes_list, "time", na_replace=0.0
    )

    # print(moving_volumes.columns)
    # print(moving_volumes.index)
    # print((moving_volumes == 0.0).sum())

    if return_first_timestamps:
        return moving_volumes, first_timestamps

    return moving_volumes


def simulate(
    params,
    moving_volumes,
    first_timestamps,
    get_buys_and_sells_fn,
    balance,
    k=20,
    assume_buy=True,
    return_separate=False,
):
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    args = params["params"]
    aggregate_N, w = args[:2]
    strategy_N = aggregate_N * w

    secondary_args = params["secondary_params"]

    balancing_period = secondary_args["balancing_period"]
    displacement = secondary_args["displacement"]
    stop_loss = secondary_args["stop_loss"]
    take_profit = secondary_args["take_profit"]
    target_usd_percentage = secondary_args["target_usd_percentage"]

    k = min(k, len(moving_volumes.columns))
    idx_k = np.arange(k)

    pnl = np.zeros((len(moving_volumes.index), k + 1))
    pnl[0, :k] = 1.0 / k

    active_symbols = np.array([""] * k)
    timestamp_i = 0  # TODO: randomize first timestamp?
    timestamp = moving_volumes.index[timestamp_i]
    next_timestamps = np.array([timestamp] * k)

    idle_count = 0

    with tqdm(
        total=int(moving_volumes.index[-1] - moving_volumes.index[0]) // 3600
    ) as progress_bar:
        while timestamp < moving_volumes.index[-1]:
            li = next_timestamps == timestamp
            active_symbols[li] = ""
            volumes = moving_volumes.loc[timestamp].sort_values(ascending=False)

            i = 0

            for active_symbol_i in idx_k[li]:
                total_pnl = np.sum(pnl[timestamp_i, :])
                if pnl[timestamp_i, active_symbol_i] > total_pnl / k:
                    pnl[timestamp_i, k] += (
                        pnl[timestamp_i, active_symbol_i] - total_pnl / k
                    )
                    pnl[timestamp_i, active_symbol_i] = total_pnl / k
                else:
                    pnl_diff = min(
                        total_pnl / k - pnl[timestamp_i, active_symbol_i],
                        pnl[timestamp_i, k],
                    )

                    pnl[timestamp_i, k] -= pnl_diff
                    pnl[timestamp_i, active_symbol_i] += pnl_diff

                while i < len(volumes.index) and active_symbols[active_symbol_i] == "":
                    symbol = volumes.index[i]
                    i += 1

                    if symbol in active_symbols:
                        continue

                    first_timestamp = first_timestamps[symbol]

                    if first_timestamp > timestamp:
                        continue

                    (
                        aggregated_closes,
                        aggregated_lows,
                        aggregated_highs,
                        aggregated_times,
                        spread_data,
                    ) = data.load_data_for_symbol(
                        data_dir, symbol, split_data=False, displacement=displacement
                    )

                    price_data_timestamp_i = utils.binary_search(
                        aggregated_times, timestamp
                    )
                    aggregated_closes_up_to_timestamp = aggregated_closes[:price_data_timestamp_i]

                    buy = assume_buy

                    if len(aggregated_closes_up_to_timestamp) > strategy_N:
                        # ignore possible gaps
                        buy, _, N = get_buys_and_sells_fn(
                            aggregated_closes_up_to_timestamp[-(strategy_N + 1) :],
                            *args,
                            as_boolean=True,
                            from_minute=False
                        )
                        assert N == 1

                    if not buy:
                        continue

                    aggregated_closes = aggregated_closes[price_data_timestamp_i - strategy_N:]
                    aggregated_lows = aggregated_lows[price_data_timestamp_i - strategy_N:]
                    aggregated_highs = aggregated_highs[price_data_timestamp_i - strategy_N:]
                    aggregated_times = aggregated_times[price_data_timestamp_i - strategy_N:]

                    if len(aggregated_closes) <= strategy_N:
                        continue

                    # ignore possible gaps
                    buys, sells, N = get_buys_and_sells_fn(
                        aggregated_closes, *args, from_minute=False
                    )
                    aggregated_closes = aggregated_closes[-(N + 1):]
                    aggregated_lows = aggregated_lows[-(N + 1):]
                    aggregated_highs = aggregated_highs[-(N + 1):]
                    aggregated_times = aggregated_times[-(N + 1):]

                    tax_exemption = utils.get_leverage_from_symbol(symbol) == 1

                    _, trade_ends = opt.get_trades(buys, sells, aggregate_N, False)

                    trade_end = trade_ends[0] if len(trade_ends) > 0 else 1

                    trade_closes = (
                        aggregated_closes[1 : trade_end + 1] / aggregated_closes[0]
                    )
                    trade_lows = (
                        aggregated_lows[1 : trade_end + 1] / aggregated_closes[0]
                    )
                    trade_highs = (
                        aggregated_highs[1 : trade_end + 1] / aggregated_closes[0]
                    )
                    trade_times = aggregated_times[1 : trade_end + 1]

                    stop_loss_index = opt.get_stop_loss_index(trade_lows, stop_loss)
                    take_profit_index = opt.get_take_profit_index(
                        trade_highs, take_profit
                    )
                    trigger_i = min(stop_loss_index, take_profit_index)

                    usd_values, asset_values, _ = opt.get_balanced_trade(
                        trade_closes,
                        stop_loss_index,
                        take_profit_index,
                        stop_loss,
                        take_profit,
                        target_usd_percentage,
                        balancing_period,
                        spread_data,
                        balance * pnl[timestamp_i, active_symbol_i],
                        taxes=True,
                        tax_exemption=tax_exemption,
                        from_minute=False,
                    )

                    trade_values = usd_values + asset_values

                    if trigger_i == len(trade_closes):
                        trigger_i -= 1

                    next_timestamp = timestamp + float(
                        utils.get_smallest_divisible_larger_than(
                            int(trade_times[trigger_i] - timestamp),
                            aggregate_N * 3600,
                        )
                    )

                    next_timestamps[active_symbol_i] = next_timestamp

                    # no need to substract 1 since these are used just with slice
                    trigger_timestamp_i = utils.binary_search(
                        moving_volumes.index.values, trade_times[trigger_i]
                    )
                    next_timestamp_i = utils.binary_search(
                        moving_volumes.index.values, next_timestamp
                    )

                    pnl[timestamp_i + 1 : trigger_timestamp_i, active_symbol_i] = (
                        np.interp(
                            moving_volumes.index.values[
                                timestamp_i + 1 : trigger_timestamp_i
                            ],
                            trade_times[: trigger_i + 1],
                            trade_values[: trigger_i + 1],
                        )
                        * pnl[timestamp_i, active_symbol_i]
                    )
                    pnl[trigger_timestamp_i:next_timestamp_i, active_symbol_i] = pnl[
                        trigger_timestamp_i - 1, active_symbol_i
                    ]

                    active_symbols[active_symbol_i] = symbol

                if active_symbols[active_symbol_i] == "":
                    idle_count += aggregate_N
                    # move next_timestamps[active_symbol_i] to next time
                    # based on aggregate_N
                    #   - the values of pnl (column active_symbol_i) from this
                    #     timestamp to the next should remain constant
                    next_timestamp = timestamp + aggregate_N * 3600.0
                    next_timestamps[active_symbol_i] = next_timestamp

                    # no need to substract 1 since this is used just with slice
                    next_timestamp_i = utils.binary_search(
                        moving_volumes.index.values, next_timestamp
                    )

                    pnl[timestamp_i + 1 : next_timestamp_i, active_symbol_i] = pnl[
                        timestamp_i, active_symbol_i
                    ]

            next_timestamp = np.min(next_timestamps)
            progress_bar.update(int(next_timestamp - timestamp) // 3600)
            timestamp = next_timestamp

            timestamp_i = (
                utils.binary_search(moving_volumes.index.values, timestamp) - 1
            )
            assert (
                timestamp >= moving_volumes.index[-1]
                or timestamp == moving_volumes.index[timestamp_i]
            )

    if not return_separate:
        pnl = np.sum(pnl, axis=1)

    return pnl, idle_count


def evaluate(
    optim_results_dir="optim_results",
    params_fname="params.json",
    whitelist_fname="whitelist.json",
    strategy_type="macross",
    k=20,
    debug=False,
):
    params_fname = os.path.join(optim_results_dir, params_fname)
    with open(params_fname, "r") as file:
        params = json.load(file)

    displacement = params["secondary_params"]["displacement"]

    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    markets = data.get_filtered_markets(client, filter_volume=True)
    markets = markets["name"]
    symbols = markets.map(lambda x: x.split("/")[0])

    whitelist_fname = os.path.join(optim_results_dir, whitelist_fname)
    with open(whitelist_fname, "r") as file:
        whitelist = json.load(file)

    li = symbols.map(lambda x: whitelist[x])
    markets = markets[li]
    symbols = symbols[li]

    moving_volumes, first_timestamps = get_moving_volumes(
        markets, displacement, k=3 if debug else None, return_first_timestamps=True
    )

    get_buys_and_sells_fn = utils.choose_get_buys_and_sells_fn(strategy_type)

    k = min(k, len(moving_volumes.columns))
    balance = utils.get_total_balance(client, False)

    pnl, idle_count = simulate(
        params,
        moving_volumes,
        first_timestamps,
        get_buys_and_sells_fn,
        balance,
        k=k,
        assume_buy=False,
        return_separate=True,
    )

    print()

    max_idle_count = k * len(moving_volumes)
    print("Fraction idle:", idle_count / max_idle_count)

    total_pnl = np.sum(pnl, axis=1)
    print("PnL:", utils.round_to_n(total_pnl[-1]))

    n_months = len(total_pnl) / (24 * 30)
    print("Number of months:", utils.round_to_n(n_months))
    print("=> PnL (monthly):", utils.round_to_n(total_pnl[-1] ** (1 / n_months), 4))

    plt.style.use("seaborn")
    plt.plot(pnl[:, :k] * k, "k", alpha=0.15)
    plt.plot(total_pnl, "g")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    evaluate(
        optim_results_dir="optim_results",
        params_fname="params.json",
        whitelist_fname="whitelist.json",
        strategy_type="macross",
        k=10,
        debug=False,
    )
