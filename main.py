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

# TODO: train a network for predicitng the optimal take profit parameter(s)

# TODO: make a generator for calculating wealths

# TODO: make a "daily" script that saves new optimal values periodically

# TODO: add comments; improve readability

# TODO: write a bunch of tests

# TODO: Automate everything

# TODO: smart print function for printing dicts



# TODO: what if stop losses were calculated from (minute time frame) closes?
#   would speed up aggregate(...) massively, if take profits were also calculated
#   from those.

# TODO: visualize weighted adaptive spreads

# TODO: make an adaptive version of plot_displacement

# TODO: only trade with .../USD

# TODO: make a maker strategy instead of taker?

# TODO: implement weighted adaptive strategies in trade.py

# TODO: implement different objective functions in find_optimal_aggregated_strategy


def calculate_optimal_take_profit(take_profit_options, trade_wealths, max_trade_wealths):
    take_profit_options.sort()
    res = np.ones((trade_wealths.shape[0],)) * take_profit_options[0]

    for take_profit in take_profit_options[1:]:
        li = max_trade_wealths > take_profit
        res[li] = take_profit

    return res

def get_something(something, entries, exits, N):
    entries_idx, _ = get_entry_and_exit_idx(entries, exits, N)

    return something[entries_idx]

# Try to predict take_profit_short based on volatility (standard deviation)
def asdf():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    aggregate_N, w, m, m_bear = (3, 16, 3, 3)
    N_repeat = 100
    randomize = True

    long_trade_wealths = []
    max_long_trade_wealths = []

    short_trade_wealths = []
    max_short_trade_wealths = []

    short_stds = []

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for X in Xs:
        X_orig = X

        X_bear = get_multiplied_X(X, -m_bear)
        if m > 1:
            X = get_multiplied_X(X, m)

        for n in tqdm(range(N_repeat), disable = N_repeat == 1):
            rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
            if rand_N > 0:
                X1 = X[:-rand_N, :]
                X1_orig = X_orig[:-rand_N, :]
                X1_bear = X_bear[:-rand_N, :]
            else:
                X1 = X
                X1_orig = X_orig
                X1_bear = X_bear
            X1_agg = aggregate(X1, aggregate_N)
            X1_orig_agg = aggregate(X1_orig, aggregate_N)
            X1_bear_agg = aggregate(X1_bear, aggregate_N)

            buys, sells, N, std = get_buys_and_sells(X1_orig_agg, w, return_std = True)

            X1_agg = X1_agg[-N:, :]
            X1_bear_agg = X1_bear_agg[-N:, :]

            wealths = get_wealths(
                X1_agg,
                buys,
                sells,
                commissions = commissions,
                spread_bull = spread if m <= 1.0 else spread_bull,
                X_bear = X1_bear_agg,
                spread_bear = spread_bear
            )

            append_trade_wealths(wealths, buys, sells, N, long_trade_wealths, max_long_trade_wealths)
            append_trade_wealths(wealths, sells, buys, N, short_trade_wealths, max_short_trade_wealths)

            short_std = get_something(std, sells, buys, N)
            short_stds.extend(short_std)

    short_trade_wealths = np.array(short_trade_wealths)
    max_short_trade_wealths = np.array(max_short_trade_wealths)
    # take_profit_options = np.array([1.69, 2.13])
    take_profit_options = max_short_trade_wealths
    # print(take_profit_options)
    optimal_take_profits = calculate_optimal_take_profit(take_profit_options, short_trade_wealths, max_short_trade_wealths)

    short_stds = np.array(short_stds)
    if take_profit_options.shape[0] <= 4:
        for take_profit in take_profit_options:
            plt.hist(
                short_stds[optimal_take_profits == take_profit],
                np.sqrt(N_repeat).astype(int) * 2,
                density = True,
                alpha = 0.9 / np.sqrt(take_profit_options.shape[0])
            )
    else:
        plt.plot(short_stds, optimal_take_profits, '.', alpha = 0.25)
    plt.show()

# Try to predict expected return based on volatility (standard deviation)
def asdf2():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    aggregate_N, w, m, m_bear, take_profit_long, take_profit_short = (3, 16, 3, 3, 1.69, 2.13)
    N_repeat = 100
    randomize = True

    long_trade_wealths = []
    max_long_trade_wealths = []

    short_trade_wealths = []
    max_short_trade_wealths = []

    short_stds = []
    long_stds = []

    Xs = load_all_data(test_files, [0, 1])

    if not isinstance(Xs, list):
        Xs = [Xs]

    for X in Xs:
        X_orig = X

        X_bear = get_multiplied_X(X, -m_bear)
        if m > 1:
            X = get_multiplied_X(X, m)

        for n in tqdm(range(N_repeat), disable = N_repeat == 1):
            rand_N = np.random.randint(aggregate_N * 60) if randomize else 0
            if rand_N > 0:
                X1 = X[:-rand_N, :]
                X1_orig = X_orig[:-rand_N, :]
                X1_bear = X_bear[:-rand_N, :]
            else:
                X1 = X
                X1_orig = X_orig
                X1_bear = X_bear
            X1_agg = aggregate(X1, aggregate_N)
            X1_orig_agg = aggregate(X1_orig, aggregate_N)
            X1_bear_agg = aggregate(X1_bear, aggregate_N)

            buys, sells, N, std = get_buys_and_sells(X1_orig_agg, w, return_std = True)

            X1_agg = X1_agg[-N:, :]
            X1_bear_agg = X1_bear_agg[-N:, :]

            wealths = get_wealths(
                X1_agg,
                buys,
                sells,
                commissions = commissions,
                spread_bull = spread if m <= 1.0 else spread_bull,
                X_bear = X1_bear_agg,
                spread_bear = spread_bear
            )

            append_trade_wealths(wealths, buys, sells, N, long_trade_wealths, max_long_trade_wealths)
            append_trade_wealths(wealths, sells, buys, N, short_trade_wealths, max_short_trade_wealths)

            short_std = get_something(std, sells, buys, N)
            short_stds.extend(short_std)

            long_std = get_something(std, buys, sells, N)
            long_stds.extend(long_std)

    long_trade_wealths = get_take_profit_wealths_from_trades(long_trade_wealths, max_long_trade_wealths, take_profit_long, 1, commissions, spread if m <= 1.0 else spread_bull, return_total = False)
    short_trade_wealths = get_take_profit_wealths_from_trades(short_trade_wealths, max_short_trade_wealths, take_profit_short, 1, commissions, spread_bear, return_total = False)

    long_stds = np.array(long_stds)
    short_stds = np.array(short_stds)

    plt.plot(long_stds, long_trade_wealths, 'g.', alpha = 0.1)
    plt.show()
    plt.plot(short_stds, short_trade_wealths, 'r.', alpha = 0.1)
    plt.show()




def wave_visualizer(params, short = True, take_profit = True):
    aggregate_N, w, m, m_bear, take_profit_long, take_profit_short = params

    x = np.arange(1, 20 * w)
    A = 1
    omega = 0.5

    X = (x / w + A * np.sin(omega * np.pi * x / w)) / 100 + 1
    X = X.reshape(-1, 1)
    X_orig = X

    if short:
        X_bear = get_multiplied_X(X, -m_bear)
    if m > 1:
        X = get_multiplied_X(X, m)

    buys, sells, N = get_buys_and_sells(X, w)
    X = X[-N:, :]
    X_orig = X_orig[-N:, :]
    X_bear = X_bear[-N:, :]

    wealths = get_wealths(
        X,
        buys,
        sells,
        commissions = commissions,
        spread_bull = spread if m <= 1.0 else spread_bull,
        X_bear = X_bear,
        spread_bear = spread_bear
    )

    if take_profit:
        transform_wealths(wealths, buys, sells, N, take_profit_long, commissions, spread if m <= 1.0 else spread_bull)
        transform_wealths(wealths, sells, buys, N, take_profit_short, commissions, spread_bear)

    plt.plot(X_orig / X_orig[0, 0])
    plt.plot(wealths)
    if np.log10(wealths[-1]) >= 1:
        plt.yscale('log')
    plt.show()



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

def orderbook_experiments(symbol, plots = True):
    client = FtxClient(ftx_api_key, ftx_secret_key)
    # total_balance = get_total_balance(client, False)
    total_balance = 1000
    # print(total_balance)
    print(symbol)

    orderbook = client.get_orderbook(symbol + '/USD', 100)
    time.sleep(0.05)
    asks = np.array(orderbook['asks'])
    bids = np.array(orderbook['bids'])
    price = (asks[0, 0] + bids[0, 0]) / 2

    usd_value_asks = np.prod(asks, axis=1)
    usd_value_bids = np.prod(bids, axis=1)
    percentage_ask = asks[:, 0] / price
    percentage_bid = bids[:, 0] / price

    # print(np.stack([usd_value_asks, percentage_ask], axis=1))

    # TODO: check edge cases
    #   - average spread is larger than the last profitable spread
    #   - total balance is larger than total available asks
    li_asks = np.cumsum(usd_value_asks) < total_balance
    weights_ask = usd_value_asks[li_asks] / total_balance
    weights_ask = np.concatenate([weights_ask, [1 - np.sum(weights_ask)]])
    print(weights_ask, np.sum(weights_ask))

    average_spread_ask = np.sum(percentage_ask[:len(weights_ask)] * weights_ask)
    print(average_spread_ask)

    if plots:
        plt.style.use('seaborn')

        plt.plot(percentage_ask, np.cumsum(usd_value_asks))
        plt.plot(percentage_bid, np.cumsum(usd_value_bids))
        plt.show()



def calculate_average_spreads():
    plt.style.use('seaborn')
    step = 0.0001
    total_balance = 628 * 0.25
    print(total_balance)

    foldernames = glob.glob('data/orderbooks/ETH/*/')

    for folder in foldernames:
        filenames = glob.glob(folder + '*.json')
        print(folder)

        distribution_ask = []
        distribution_bid = []

        for filename in filenames:
            with open(filename, 'r') as fp:
                orderbook = json.load(fp)

            asks = np.array(orderbook['asks'])
            bids = np.array(orderbook['bids'])
            price = (asks[0, 0] + bids[0, 0]) / 2

            usd_value_asks = np.prod(asks, axis=1)
            usd_value_bids = np.prod(bids, axis=1)
            percentage_ask = asks[:, 0] / price - 1
            percentage_bid = bids[:, 0] / price - 1

            while len(distribution_ask) * step < np.max(percentage_ask) + step:
                distribution_ask.append(0.0)

            idx_ask = np.ceil(percentage_ask / step).astype(int)

            distribution_ask = np.array(distribution_ask)
            distribution_ask[idx_ask] += usd_value_asks / len(filenames)
            distribution_ask = list(distribution_ask)

            while -len(distribution_bid) * step > np.min(percentage_bid) - step:
                distribution_bid.append(0.0)

            idx_bid = np.ceil(np.abs(percentage_bid) / step).astype(int)

            distribution_bid = np.array(distribution_bid)
            distribution_bid[idx_bid] += usd_value_bids / len(filenames)
            distribution_bid = list(distribution_bid)

        distribution_ask = np.array(distribution_ask)
        percentage_ask = np.linspace(0, len(distribution_ask) * step, len(distribution_ask))

        li_asks = np.cumsum(distribution_ask) < total_balance
        weights_ask = distribution_ask[li_asks] / total_balance
        weights_ask = np.concatenate([weights_ask, [1 - np.sum(weights_ask)]])
        average_spread_ask = np.sum(percentage_ask[:len(weights_ask)] * weights_ask)
        print(average_spread_ask)

        distribution_bid = np.array(distribution_bid)
        percentage_bid = np.linspace(0, -len(distribution_bid) * step, len(distribution_bid))

        li_bids = np.cumsum(distribution_bid) < total_balance
        weights_bid = distribution_bid[li_bids] / total_balance
        weights_bid = np.concatenate([weights_bid, [1 - np.sum(weights_bid)]])
        average_spread_bid = np.sum(percentage_bid[:len(weights_bid)] * weights_bid)
        print(average_spread_bid)

        plt.plot(percentage_ask, distribution_ask)
        plt.plot(percentage_bid, distribution_bid)
        plt.show()


def test_calc_aggreagte_buys_and_sells():
    aggregate_N_list = [3]
    w_list = [16]
    m = m_bear = 3
    short = True

    files = glob.glob(f'data/ETH/*.json')
    files.sort(key = get_time)

    Xs = load_all_data(files, 0)

    if not isinstance(Xs, list):
        Xs = [Xs]

    X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    X_origs = Xs
    if m > 1:
        Xs = [get_multiplied_X(X, m) for X in Xs]

    for aggregate_N, w in tqdm(list(product(aggregate_N_list, w_list)), disable = True):

        t1 = time.time()
        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            buys_orig, sells_orig, N_orig = get_buys_and_sells_ma(X_orig, aggregate_N * 60 * w)

            for rand_N in range(aggregate_N * 60):
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
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths_fast(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )
        t2 = time.time()
        print(round_to_n(t2 - t1, 3))
        t1 = time.time()
        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            for rand_N in range(aggregate_N * 60):
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    X1_orig = X_orig[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    X1_orig = X_orig
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                X1_orig_agg = aggregate(X1_orig, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells_ma(X1_orig_agg, w)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths_fast(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )
        t2 = time.time()
        print(round_to_n(t2 - t1, 3))
        t1 = time.time()
        for X_orig, X, X_bear in zip(X_origs, Xs, X_bears):
            for rand_N in range(aggregate_N * 60):
                if rand_N > 0:
                    X1 = X[:-rand_N, :]
                    X1_orig = X_orig[:-rand_N, :]
                    if short:
                        X1_bear = X_bear[:-rand_N, :]
                else:
                    X1 = X
                    X1_orig = X_orig
                    if short:
                        X1_bear = X_bear
                X1_agg = aggregate(X1, aggregate_N)
                X1_orig_agg = aggregate(X1_orig, aggregate_N)
                if short:
                    X1_bear_agg = aggregate(X1_bear, aggregate_N)

                buys, sells, N = get_buys_and_sells(X1_orig_agg, w)

                X1_agg = X1_agg[-N:, :]
                if short:
                    X1_bear_agg = X1_bear_agg[-N:, :]
                X1 = X1[-aggregate_N * 60 * N:, :]
                if short:
                    X1_bear = X1_bear[-aggregate_N * 60 * N:, :]

                wealths = get_wealths_fast(
                    X1_agg,
                    buys,
                    sells,
                    commissions = commissions,
                    spread_bull = spread if m <= 1.0 else spread_bull,
                    X_bear = X1_bear_agg,
                    spread_bear = spread_bear
                )
        t2 = time.time()
        print(round_to_n(t2 - t1, 3))


def simple_objective(x):
    return x


def main():
    # client = FtxClient(ftx_api_key, ftx_secret_key)
    # coin = 'ETH'
    # m = 3
    # m_bear = 3
    #
    # total_balance = get_total_balance(client, False)
    # total_balance = max(total_balance, 1)
    # potential_balances = np.logspace(np.log10(total_balance / 10), np.log10(total_balance * 1000), 2000)
    # potential_spreads = get_average_spread(coin, m, potential_balances, m_bear = m_bear)
    #
    # files = glob.glob(f'data/{coin}/*.json')
    # files.sort(key = get_time)
    #
    # Xs = load_all_data(files, [0, 1])
    #
    # if not isinstance(Xs, list):
    #     Xs = [Xs]
    #
    # X_bears = [get_multiplied_X(X, -m_bear) for X in Xs]
    # X_origs = Xs
    # if m > 1:
    #     Xs = [get_multiplied_X(X, m) for X in Xs]
    #
    # params_dict = get_objective_function(
    #         args = (10, 44),
    #         strategy_type = 'ma',
    #         frequency = 'low',
    #         N_repeat_inp = 40,
    #         sides = ['long', 'short'],
    #         X_origs = X_origs,
    #         Xs = Xs,
    #         X_bears = X_bears,
    #         short = True,
    #         step = 0.01,
    #         stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing'],
    #         total_balance = total_balance,
    #         potential_balances = potential_balances,
    #         potential_spreads = potential_spreads,
    #         workers = 4,
    #         debug = True
    # )
    #
    # for key, value in params_dict.items():
    #     print(key)
    #     print(value)

    # parameter_names = ['aggregate_N', 'w']

    all_bounds = {
        'aggregate_N': (1, 12),
        'w': (1, 50),
        'w2': (1, 20),
    }

    all_resolutions = {
        'aggregate_N': 1,
        'w': 1,
        'w2': 1,
    }

    # init_args = (10, 44)

    # args = np.array((3, 16))
    # resolution_values = np.array([v for v in resolutions.values()]).reshape(1, -1)
    # ds = np.array(list(product([-1, 0, 1], repeat = len(bounds))))
    # candidate_args = ds * resolution_values + args.reshape(1, -1)
    # for i in range(len(bounds)):
    #     candidate_args[:, i] = np.clip(candidate_args[:, i], *bounds[parameter_names[i]])
    # candidate_args = np.unique(candidate_args, axis=0)
    #
    # print(candidate_args)


    save_optimal_parameters(
        all_bounds = all_bounds,
        all_resolutions = all_resolutions,
        coins = ['ETH', 'BTC'],
        frequencies = ['low', 'high'],
        strategy_types = ['ma', 'ma_cross'],
        stop_loss_take_profit_types = ['stop_loss', 'take_profit', 'trailing'],
        N_iter = 50,
        m = 3,
        m_bear = 3,
        N_repeat_inp = 40,
        step = 0.01,
        verbose = True,
        disable = False,
        short = True,
        Xs_index = [0, 1],
        debug = False
    )


    # a = np.arange(3)
    # b = np.arange(3) + 4
    #
    # print(a)
    # print(b)
    # c = np.stack([a, b], axis = -1)
    # print(c)
    # print(c.flatten())

    # client = FtxClient(ftx_api_key, ftx_secret_key)
    # total_balance = get_total_balance(client, False) / 4
    # total_balances = np.logspace(np.log10(total_balance / 10), np.log10(total_balance * 100), 100)
    # spreads = get_average_spread('ETH', 3, total_balances, m_bear = 3)
    # spread = spreads[np.argmin(np.abs(total_balances - total_balance))]
    # print(spread)
    # plt.style.use('seaborn')
    # plt.plot(total_balances, spreads)
    # plt.show()

    # strategies = {}
    # coins = ['ETH', 'BTC']
    # freqs = ['low', 'high']
    # strategy_types = ['ma']
    #
    # for coin, freq, strategy_type in product(coins, freqs, strategy_types):
    #     strategy_key = '_'.join([coin, freq, strategy_type])
    #     filename = f'optim_results/{strategy_key}.json'
    #     with open(filename, 'r') as file:
    #         strategies[strategy_key] = json.load(file)

    # with open('optim_results/weights.json', 'r') as file:
    #     weights = json.load(file)
    #
    # get_adaptive_wealths_for_multiple_strategies(
    #     strategies = strategies,
    #     m = 3,
    #     m_bear = 3,
    #     weights = weights,
    #     N_repeat_inp = 5,
    #     disable = False,
    #     verbose = True,
    #     randomize = True,
    #     Xs_index = [0, 1]
    # )

    # plot_weighted_adaptive_wealths(
    #     strategies,
    #     m = 3,
    #     m_bear = 3,
    #     N_repeat = 5,
    #     short = True,
    #     randomize = True,
    #     Xs_index = [0, 1]
    # )

    # optimize_weights_iterative(
    #     coins = ['ETH', 'BTC'],
    #     freqs = ['low', 'high'],
    #     strategy_types = ['ma'],
    #     weights_type = "file",
    #     n_iter = 2
    # )

    # test_calc_aggreagte_buys_and_sells()

    # calculate_average_spreads()

    # asdf2()
    # wave_visualizer((3, 16, 3, 3, 1.69, 2.13))

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
