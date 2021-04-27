import numpy as np
import keys
from ftx.rest.client import FtxClient
from algtra.collect import data
import os
from algtra import constants, utils
import algtra.optimize.utils as opt
import matplotlib.pyplot as plt
from datetime import datetime
from ciso8601 import parse_datetime
import time
from tqdm import tqdm
import pandas as pd


def main():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    markets = data.get_filtered_markets(client, filter_volume=True)
    markets = markets["name"]

    volumes_list = []
    displacement = 55

    for market in tqdm(markets):
        price_data = data.load_price_data(data_dir, market, return_price_data_only=True)
        if price_data is not None:
            symbol = market.split("/")[0]

            times = price_data["time"].values
            volumes = price_data["volume"].values

            first_time = price_data["startTime"][0]
            first_timestamp = datetime.timestamp(parse_datetime(first_time))
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
                moving_daily_volumes, times = utils.gap_aware_moving_sum(
                    aggregated_volumes, aggregated_times, window_size=24
                )

                # print(int(times[0]), int(times[-1]), int(times[-1] - times[0]) // 3600)

                volumes_list.append(
                    pd.DataFrame(
                        {
                            symbol: moving_daily_volumes,
                            "time": times,
                        }
                    )
                )

    moving_daily_volumes = utils.combine_time_series_by_common_time(
        volumes_list, "time", na_replace=0.0
    )

    print(moving_daily_volumes.columns)
    print(moving_daily_volumes.index)
    # print((moving_daily_volumes == 0.0).sum())


def main2():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )
    symbols = opt.get_symbols_for_optimization(k=25, min_length=250000)
    price_data = {}
    spread_data = {}

    for symbol in tqdm(symbols):
        symbol_price_data, symbol_spread_data = data.load_data_for_symbol(
            data_dir, symbol, split_data=False
        )

        price_data[symbol] = symbol_price_data
        spread_data[symbol] = symbol_spread_data

    # symbol = "BTC"

    args = (23, 60, 2)
    # price_data = {
    #     symbol: data.load_price_data(
    #         data_dir, symbol + "/USD", return_price_data_only=True
    #     )
    # }
    stop_loss = 0.2384283270784175
    take_profit = 74.17749133263747
    target_usd_percentage = 0.031753087736335506
    balancing_period = 47
    displacement = 14
    get_buys_and_sells_fn = utils.get_buys_and_sells_macross
    # spread_data = {symbol: data.load_spread_distributions(data_dir, symbol, stack=True)}
    balance = 2800.0 / 20
    use_all_start_times = True
    quantile = 0.4
    init_sub_args = None
    init_points = 100
    min_avg_monthly_return = 1.05
    min_n_samples = 10

    # optimized_params, objective, stats = opt.optimize_sltp_and_balancing_params(
    #     init_sub_args,
    #     init_points,
    #     args,
    #     price_data,
    #     get_buys_and_sells_fn,
    #     spread_data,
    #     balance,
    #     use_all_start_times=use_all_start_times,
    #     quantile=quantile,
    #     min_avg_monthly_return=min_avg_monthly_return,
    #     debug=True,
    # )
    # print(objective)
    # utils.print_dict(optimized_params)
    # utils.print_dict(stats)

    t0 = time.perf_counter()

    (
        objective,
        median,
        mean,
        fraction_of_positives,
    ) = opt.get_sltp_and_balancing_objective_function(
        args=args,
        price_data=price_data,
        stop_loss=stop_loss,
        take_profit=take_profit,
        target_usd_percentage=target_usd_percentage,
        balancing_period=balancing_period,
        displacement=displacement,
        get_buys_and_sells_fn=get_buys_and_sells_fn,
        spread_data=spread_data,
        balance=balance,
        use_all_start_times=use_all_start_times,
        quantile=quantile,
        min_avg_monthly_return=min_avg_monthly_return,
        min_n_samples=min_n_samples,
    )
    t1 = time.perf_counter()
    print(f"Done (took {utils.round_to_n((t1 - t0) / 1)} seconds).")
    print(objective, median, mean, fraction_of_positives)
    print()

    t0 = time.perf_counter()

    (
        objective,
        median,
        mean,
        fraction_of_positives,
    ) = opt.get_sltp_and_balancing_objective_function(
        args=args,
        price_data=price_data,
        stop_loss=stop_loss,
        take_profit=take_profit,
        target_usd_percentage=target_usd_percentage,
        balancing_period=balancing_period,
        displacement=displacement,
        get_buys_and_sells_fn=get_buys_and_sells_fn,
        spread_data=spread_data,
        balance=balance,
        use_all_start_times=use_all_start_times,
        quantile=quantile,
        min_avg_monthly_return=min_avg_monthly_return,
        min_n_samples=min_n_samples,
    )
    t1 = time.perf_counter()
    print(f"Done (took {utils.round_to_n((t1 - t0) / 1)} seconds).")
    print(objective, median, mean, fraction_of_positives)
    print()


if __name__ == "__main__":
    main()
