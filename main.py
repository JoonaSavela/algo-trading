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
from parameters import commissions, minutes_in_a_year
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
from ciso8601 import parse_datetime
from collections import deque


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


def get_all_price_data(client, market):
    res = []

    X = client.get_historical_prices(
        market = market,
        resolution = 60,
        limit = 5000,
    )
    time.sleep(0.05)
    res.extend(X)
    count = 1
    start_time = min(parse_datetime(x['startTime']) for x in X)
    print(count, len(X), start_time)
    start_time = start_time.timestamp()

    while len(X) >= 5000:
        X = client.get_historical_prices(
            market = market,
            resolution = 60,
            limit = 5000,
            end_time = start_time
        )
        time.sleep(0.05)
        res.extend(X)
        count += 1
        start_time = min(parse_datetime(x['startTime']) for x in X)
        print(count, len(X), start_time)
        start_time = start_time.timestamp()

    res = pd.DataFrame(res).drop_duplicates('startTime').sort_values('startTime')

    return res



def get_buy_value_from_buy_history(buy_history, sell_size, verbose = False):
    buy_prices = []
    buy_sizes = []

    while sell_size > 0 and buy_history:
        buy_price, buy_size = buy_history.popleft()

        size_diff = min(sell_size, buy_size)
        sell_size -= size_diff
        buy_size -= size_diff

        buy_prices.append(buy_price)
        buy_sizes.append(size_diff)

        if buy_size > 0:
            buy_history.appendleft((buy_price, buy_size))

    if verbose:
        print(buy_history)
        print(buy_prices)
        print(buy_sizes)

    return np.dot(buy_prices, buy_sizes)



def main():


    # (price, size)
    # buy1 = (1234.0, 0.2)
    # buy2 = (1534.1, 0.36)
    # buy3 = (1434.1, 0.1)
    #
    # symbol = 'ETHBULL'
    # buy_history = {
    #     symbol: deque()
    # }
    #
    # buy_history[symbol].append(buy1)
    # buy_history[symbol].append(buy2)
    # buy_history[symbol].append(buy3)
    #
    # print(buy_history)
    #
    # sell_price, sell_size = (2643.12, 0.3)
    # sell_value = sell_price * sell_size
    # print(sell_value)
    #
    # buy_value = get_buy_value_from_buy_history(buy_history[symbol], sell_size, verbose = True)
    #
    # print(buy_history)
    # print(buy_value)
    # profit = sell_value / buy_value
    # print(profit)





    # m = 3
    # m_bear = 3
    # short = True
    # coin = 'ETH'
    # N_repeat_inp = 10
    # place_take_profit_and_stop_loss_simultaneously = True
    # workers = None
    # stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing']
    # # stop_loss_take_profit_types = ['take_profit',]
    #
    # files = glob.glob(f'data/{coin}/*.json')
    # files.sort(key = get_time)
    # Xs = load_all_data(files, [0, 1])
    #
    # if not isinstance(Xs, list):
    #     Xs = [Xs]
    #
    # N_years = max(len(X) for X in Xs) / minutes_in_a_year
    #
    # if short:
    #     X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    # else:
    #     X_bears = [None] * len(Xs)
    # X_origs = Xs
    # if m > 1:
    #     Xs = [get_multiplied_X(X, m) for X in Xs]
    #
    # client = FtxClient(ftx_api_key, ftx_secret_key)
    # total_balance = get_total_balance(client, False)
    # total_balance = max(total_balance, 1)
    #
    # potential_balances = np.logspace(np.log10(total_balance / (10 * N_years)), \
    #     np.log10(total_balance * 1000 * N_years), int(4000 * N_years))
    # potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)
    #
    # args = (2, 44, 7)
    # print(args)
    # params_dict = get_objective_function(
    #     args = args,
    #     strategy_type = 'macross',
    #     frequency = 'high',
    #     coin = coin,
    #     m = m,
    #     m_bear = m_bear,
    #     N_repeat_inp = N_repeat_inp,
    #     sides = ['long', 'short'],
    #     X_origs = X_origs,
    #     Xs = Xs,
    #     X_bears = X_bears,
    #     short = short,
    #     step = 0.01,
    #     stop_loss_take_profit_types = stop_loss_take_profit_types,
    #     total_balance = total_balance,
    #     potential_balances = potential_balances,
    #     potential_spreads = potential_spreads,
    #     place_take_profit_and_stop_loss_simultaneously = place_take_profit_and_stop_loss_simultaneously,
    #     trail_value_recalc_period = None,
    #     workers = workers,
    #     debug = True
    # )
    # print(params_dict)
    # print()
    #
    # args = (12, 39, 25)
    # print(args)
    # params_dict = get_objective_function(
    #     args = args,
    #     strategy_type = 'macross',
    #     frequency = 'low',
    #     coin = coin,
    #     m = m,
    #     m_bear = m_bear,
    #     N_repeat_inp = N_repeat_inp,
    #     sides = ['long', 'short'],
    #     X_origs = X_origs,
    #     Xs = Xs,
    #     X_bears = X_bears,
    #     short = short,
    #     step = 0.01,
    #     stop_loss_take_profit_types = stop_loss_take_profit_types,
    #     total_balance = total_balance,
    #     potential_balances = potential_balances,
    #     potential_spreads = potential_spreads,
    #     place_take_profit_and_stop_loss_simultaneously = place_take_profit_and_stop_loss_simultaneously,
    #     trail_value_recalc_period = None,
    #     workers = workers,
    #     debug = True
    # )
    # print(params_dict)
    # print()


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


    client = FtxClient(ftx_api_key, ftx_secret_key)
    #
    # X = get_all_price_data(
    #     client = client,
    #     market = 'ETHBULL/USD'
    # )
    #
    # print(X)
    # print(len(X))
    # print(len(X['startTime'].unique()))

    X = client.get_historical_prices(
        market = 'ETHBULL/USD',
        resolution = 60,
        limit = 1,
        end_time = 1608472680,
        start_time = 1608472680
    )
    print(X)

    #
    # X = pd.DataFrame(X)
    #
    # # X, t = get_recent_data('ETH', size = 50, type = 'm', aggregate = 1)
    # #
    # # print(X)
    #
    # t = X['startTime'].max()[:19]
    # print(X['startTime'].map(lambda x: x[:19]))
    # print(X.tail())
    # print(t)
    # print(datetime.fromtimestamp(time.time()))
    # t = time.mktime(datetime.strptime(
    #     t,
    #     "%Y-%m-%dT%H:%M:%S"
    # ).timetuple())
    # print(t)
    # print()
    #
    # t_diff = time.time() - t - 2 * 3600 - 60
    # print(t_diff)

    # if t_diff > 0:
    #     time.sleep(60 - t_diff)
    #
    #     X = client.get_historical_prices(
    #         market = 'ETH/USD',
    #         resolution = 60,
    #         limit = 50,
    #     )
    #
    #     X = pd.DataFrame(X)
    #
    #     t = X['startTime'].max()[:19]
    #     print(X['startTime'].map(lambda x: x[:19]))
    #     print(X.tail())
    #     print(t)
    #     print(datetime.fromtimestamp(time.time()))
    #     t = time.mktime(datetime.strptime(
    #         t,
    #         "%Y-%m-%dT%H:%M:%S"
    #     ).timetuple())
    #     print(t)
    #     print()
    #
    #     t_diff = time.time() - t - 2 * 3600 - 60
    #     print(t_diff)

    # print(X)
    #
    # plt.style.use('seaborn')
    # plt.plot(X['close'])
    # plt.show()

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

    # visualize_spreads()


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
    #     place_take_profit_and_stop_loss_simultaneously = True,
    #     trail_value_recalc_period = None,
    #     randomize = False,
    #     Xs_index = [0, 1],
    #     active_balancing = False
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
