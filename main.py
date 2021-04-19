import numpy as np
import keys
from ftx.rest.client import FtxClient
from algtra.collect import data
import os
from algtra import constants, utils
import algtra.optimize.utils as opt
import matplotlib.pyplot as plt
from ciso8601 import parse_datetime
import time


def main():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )
    symbol = "ETH"

    args = (12, 41, 24)
    price_data = {
        symbol: data.load_price_data(
            data_dir, symbol + "/USD", return_price_data_only=True
        )
    }
    stop_loss = 0.0
    take_profit = 10.0
    target_usd_percentage = 0.2
    balancing_period = 12
    displacement = 55
    get_buys_and_sells_fn = utils.get_buys_and_sells_macross
    spread_data = {symbol: data.load_spread_distributions(data_dir, symbol, stack=True)}
    balance = 2800.0
    use_all_start_times = False
    quantile = 0.45
    init_sub_args = None
    init_points = 100

    # optimized_params, objective, stats = opt.optimize_sltp_and_balancing_params(
    #     init_sub_args,
    #     init_points,
    #     args,
    #     price_data,
    #     get_buys_and_sells_fn,
    #     spread_data,
    #     balance,
    #     use_all_start_times=False,
    #     quantile=0.45,
    #     min_avg_monthly_return=1.10,
    #     debug=True,
    # )
    # print(objective)
    # utils.print_dict(optimized_params)
    # utils.print_dict(stats)

    objective, median, mean, fraction_of_positives = opt.get_sltp_and_balancing_objective_function(
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
    )

    print(objective, median, mean, fraction_of_positives)


if __name__ == "__main__":
    main()
