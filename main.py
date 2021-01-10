import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
import numpy as np
import pandas as pd
import time
from datetime import datetime
from utils import *
from optimize import *
from tqdm import tqdm
from parameter_search import *
from parameters import commissions
from itertools import product
import matplotlib.animation as animation
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from trade import ftx_price, get_total_balance, get_conditional_orders, balance_portfolio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, black_litterman, \
    objective_functions
from pypfopt import BlackLittermanModel
import multiprocessing


# TODO: add comments; improve readability

# TODO: write a bunch of tests

# TODO: Automate everything

# TODO: infer total balance from deposits, withdrawals, and trades

# TODO: check that get_recent_data works correctly

# TODO: make a maker strategy instead of taker?

# TODO: implement different objective functions in find_optimal_aggregated_strategy?




def try_rounding_hours():
    seconds = time.time()
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    print(now_string)
    timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
    print(timeTo)
    print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(timeTo)))
    print()

    seconds = timeTo + 45 * 60
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    print(now_string)

    seconds += (59 - 44) * 60
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
    print(timeTo)
    print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(timeTo)))


def visualize_spreads(coin = 'ETH', m = 1, m_bear = 1):
    client = FtxClient(ftx_api_key, ftx_secret_key)
    total_balance = get_total_balance(client, False)
    total_balances = np.logspace(np.log10(total_balance / 10), np.log10(total_balance * 100), 100)
    spreads = get_average_spread(coin, m, total_balances, m_bear)
    spread = spreads[np.argmin(np.abs(total_balances - total_balance))]
    print(spread)
    plt.style.use('seaborn')
    plt.plot(total_balances, spreads)
    plt.plot([total_balance], [spread], '.k')
    plt.show()



def main():
    # wealth = 3.
    # buy_price = 1.
    # sell_price = 2.
    # profit = sell_price / buy_price
    # profit = np.array([profit])
    # print(profit, profit.shape)
    # wealth *= profit
    #
    # print(wealth)
    #
    # wealth *= apply_taxes(profit, copy = True) / profit
    # print(wealth)
    # print(3 * 1.66)

    # trades_filename = 'trading_logs/trades.csv'
    # conditional_trades_filename = 'trading_logs/conditional_trades.csv'
    #
    # trades = pd.read_csv(trades_filename, index_col = 0)
    # conditional_trades = pd.read_csv(conditional_trades_filename, index_col = 0)
    #
    # for c in trades.columns:
    #     print(c, trades[c].unique())
    #     print()
    #
    # print()
    # print()
    #
    # for c in conditional_trades.columns:
    #     print(c, conditional_trades[c].unique())
    #     print()
    #
    # ids = trades['id'].values.reshape(-1, 1)
    # ids1 = conditional_trades['orderId'].values.reshape(1, -1)
    #
    # same_ids = ids == ids1
    # print(same_ids.shape)
    # print(np.sum(same_ids, axis=0))
    # print(conditional_trades['avgFillPrice'][np.sum(same_ids, axis=0) == 0] \
    #     * conditional_trades['filledSize'][np.sum(same_ids, axis=0) == 0])
    # print(conditional_trades[np.sum(same_ids, axis=0) == 0].transpose())

    # plt.style.use('seaborn')
    # plt.spy(same_ids)
    # plt.show()


    # client = FtxClient(ftx_api_key, ftx_secret_key)
    #
    # trade_history, conditional_trade_history = get_trade_history(client)
    # print(trade_history)
    # print(conditional_trade_history)


    # end_time = time.time()
    # order_history = client.get_conditional_order_history(end_time = end_time)
    # time.sleep(0.1)
    # order_history = pd.DataFrame(order_history)
    # print(order_history['filledSize'].values)
    # print(order_history['status'].values)
    #
    # for c in order_history.columns:
    #     print(c, order_history[c].unique())
    #     print()
    #
    #
    # time_string_format = "%Y-%m-%dT%H:%M:%S"
    # timestamps = order_history['createdAt'].map(
    #     lambda x: time.mktime(
    #         datetime.strptime(
    #             x[:len(time_string_format) + 2],
    #             time_string_format
    #         ).timetuple()
    #     )
    # ) # '2021-01-03T03:54:07.975516+00:00'
    # # print(timestamps)
    #
    # order_history1 = client.get_conditional_order_history(end_time = timestamps.min())
    # order_history1 = pd.DataFrame(order_history1)
    #
    # ids = order_history['id'].values.reshape(-1, 1)
    # ids1 = order_history1['id'].values.reshape(1, -1)
    # same_id = ids == ids1
    # print(np.sum(same_id))

    # time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple())

    # print(order_history.columns)
    # print(order_history['remainingSize'].unique())
    # open_trigger_orders = get_conditional_orders(client)
    # print_dict(open_trigger_orders)


    # all_bounds = {
    #     'aggregate_N': (1, 12),
    #     'w': (1, 50),
    #     'w2': (1, 20),
    # }
    #
    # all_resolutions = {
    #     'aggregate_N': 1,
    #     'w': 1,
    #     'w2': 1,
    # }

    # save_optimal_parameters(
    #     all_bounds = all_bounds,
    #     all_resolutions = all_resolutions,
    #     coins = ['ETH', 'BTC'],
    #     frequencies = ['low', 'high'],
    #     strategy_types = ['ma', 'macross'],
    #     stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing'],
    #     N_iter = 50,
    #     m = 3,
    #     m_bear = 3,
    #     N_repeat_inp = 40,
    #     step = 0.01,
    #     skip_existing = False,
    #     verbose = True,
    #     disable = False,
    #     short = True,
    #     Xs_index = [0, 1],
    #     debug = False
    # )

    visualize_spreads()


    # coins = ['ETH', 'BTC']
    # freqs = ['low', 'high']
    # strategy_types = ['ma', 'macross']

    # strategies, weights = get_filtered_strategies_and_weights(coins, freqs, strategy_types)
    #
    # get_adaptive_wealths_for_multiple_strategies(
    #     strategies = strategies,
    #     m = 3,
    #     m_bear = 3,
    #     weights = weights,
    #     N_repeat_inp = 3,
    #     compress = 60,#1440,
    #     disable = False,
    #     verbose = True,
    #     randomize = False,
    #     Xs_index = [0, 1],
    #     debug = True
    # )

    # plot_weighted_adaptive_wealths(
    #     coins = ['ETH', 'BTC'],
    #     freqs = ['low', 'high'],
    #     strategy_types = ['ma', 'macross'],
    #     ms = [1, 3],
    #     m_bears = [0, 3],
    #     N_repeat = 1,
    #     compress = None,
    #     trail_value_recalc_period = None,
    #     randomize = True,
    #     Xs_index = [0, 1]
    # )

    # optimize_weights(compress = 60, save = True, verbose = True)

    # optimize_weights_iterative(
    #     coins = ['ETH', 'BTC'],
    #     freqs = ['low', 'high'],
    #     strategy_types = ['ma', 'macross'],
    #     weights_type = "equal",
    #     n_iter = 10,
    #     compress = 60
    # )

    # displacements = get_displacements(
    #     coins = ['ETH', 'BTC'],
    #     strategy_types = ['ma', 'macross'],
    #     ms = [1, 3],
    #     m_bears = [0, 3],
    #     sep = 2,
    #     Xs_index = [0, 1],
    #     plot = True,
    #     verbose = True,
    #     disable = False
    # )





if __name__ == "__main__":
    main()
