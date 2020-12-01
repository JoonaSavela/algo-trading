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
from parameters import commissions, spread, spread_bear, spread_bull
from itertools import product
import matplotlib.animation as animation
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from trade import ftx_price, get_total_balance
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, black_litterman, \
    objective_functions
from pypfopt import BlackLittermanModel
import multiprocessing


# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: add comments; improve readability

# TODO: write a bunch of tests

# TODO: Automate everything





# TODO: only trade with .../USD

# TODO: make a maker strategy instead of taker?

# TODO: implement weighted adaptive strategies in trade.py

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


def main():
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
    #
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

    # strategies, weights = get_filtered_strategies_and_weights(
    #     coins = ['ETH', 'BTC'],
    #     freqs = ['low', 'high'],
    #     strategy_types = ['ma', 'macross']
    # )
    #
    # weight_values = np.array(list(weights.values()))
    # print(weight_values)
    #
    # combinations = np.array(list(product([0, 1], repeat=len(weights))))
    #
    # X_diff = np.random.rand(5, len(weights)) * 0.2 + 0.9
    #
    # print(X_diff)
    # # print(np.matmul(X_diff, weight_values))
    # new_weights = X_diff * weight_values.reshape(1, -1)
    # print(new_weights)
    # print()
    #
    # for i in range(new_weights.shape[0]):
    #     new_weights_relative = new_weights[i] / np.sum(new_weights[i])
    #     print(weight_values)
    #     print(new_weights_relative)
    #     print(weight_values - new_weights_relative)
    #     print(np.allclose(np.sum(weight_values - new_weights_relative), 0))
    #
    #     # combination_i = np.random.choice(combinations.shape[0], size=1)
    #     # combination = combinations[combination_i].reshape(-1)
    #     # print(combination)
    #     # new_weights1 = np.dot(new_weights[i], combination)
    #     # weights1 = np.dot(weight_values, combination)
    #     # new_weights0 = np.dot(new_weights[i], 1 - combination)
    #     # weights0 = np.dot(weight_values, 1 - combination)
    #     #
    #     # weights_arr = np.array([weights1, weights0])
    #     # new_weights_arr = np.array([new_weights1, new_weights0])
    #     # weights_arr_relative = weights_arr / np.sum(weights_arr)
    #     # new_weights_arr_relative = new_weights_arr / np.sum(new_weights_arr)
    #     # print(weights_arr_relative)
    #     # print(new_weights_arr_relative)
    #     # print(weights_arr_relative - new_weights_arr_relative)

    plot_weighted_adaptive_wealths(
        coins = ['ETH', 'BTC'],
        freqs = ['low', 'high'],
        strategy_types = ['ma', 'macross'],
        ms = [3],
        m_bears = [3],
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
    #     m = 3,
    #     m_bear = 3,
    #     sep = 2,
    #     Xs_index = [0, 1],
    #     plot = True,
    #     verbose = True,
    #     disable = False
    # )



    # plot_performance([
    #                     # (3, 16, 1, 3, 1.2, np.Inf, 0, 0.98),
    #                     # (3, 16, 3, 3, 1.79, np.Inf, 0, 0.98), # best
    #                     # (3, 16, 3, 3, 1.79, 2.13, 0, 0),
    #                     # (10, 44, 3, 3, 4.36, 1.83, 0, 0),
    #                     (3, 16, 3, 3, 1.80, np.Inf, 0, 0.98),
    #                   ],
    #                   N_repeat = 300,
    #                   short = True,
    #                   take_profit = True,
    #                   stop_loss = True,
    #                   Xs_index = [0, 1],
    #                   N_ts_plots = 5,
    #                   last_N_to_plot = None)#90 * 24 * 60)

    # plot_displacement([
    #                     # (3, 16, 1, 3, 1.2, np.Inf, 0, 0.98),
    #                     (3, 16, 3, 3, 1.79, 2.13, 0, 0), # best
    #                   ])





if __name__ == "__main__":
    main()
