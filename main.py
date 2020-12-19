import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
import numpy as np
import pandas as pd
import time
from datetime import datetime
from utils import *
from model import *
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


# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: add comments; improve readability

# TODO: write a bunch of tests

# TODO: Automate everything



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


def visualize_spreads(coin = 'ETH', m = 3, m_bear = 3):
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


def test(buy_info):
    buy_info.loc['ETH_low_macross_3_3', 'trigger_order_id'] = '123'

def main():
    # client = FtxClient(ftx_api_key, ftx_secret_key)
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

    plot_weighted_adaptive_wealths(
        coins = ['ETH', 'BTC'],
        freqs = ['low', 'high'],
        strategy_types = ['ma', 'macross'],
        ms = [1, 3],
        m_bears = [0, 3],
        N_repeat = 3,
        compress = None,
        randomize = True,
        Xs_index = [0, 1]
    )

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
