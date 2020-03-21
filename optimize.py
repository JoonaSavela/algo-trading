from scipy.optimize import minimize, LinearConstraint, Bounds
from data import load_all_data
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from utils import *
import os
from tqdm import tqdm
import copy

# TODO: move to a different file
def get_multiplied_X(X, multiplier = 1):
    returns = X[1:, 0] / X[:-1, 0] - 1
    returns = multiplier * returns
    assert(np.all(returns > -1.0))

    X_res = np.zeros(X.shape)
    X_res[:, 0] = np.concatenate([
        [1.0],
        np.cumprod(returns + 1)
    ])

    other_returns = X[:, 1:4] / X[:, 0].reshape((-1, 1)) - 1
    other_returns = multiplier * other_returns
    assert(np.all(other_returns > -1.0))
    X_res[:, 1:4] = X_res[:, 0].reshape((-1, 1)) * (other_returns + 1)

    X_res[:, 4:] = X[:, 4:]

    return X_res

def get_wealths(X, buys, sells = None, commissions = 0.00075, short = False):
    if sells is None:
        sells = 1 - buys
    capital_usd = 1000
    capital_coin = 0
    capital_coin_bear = 0

    multiplier = 3

    price_bear = 0.0
    if short:
        X_bear = get_multiplied_X(X, -multiplier)

    X = get_multiplied_X(X, multiplier)

    wealths = [0] * X.shape[0]

    for i in range(X.shape[0]):
        BUY, SELL = buys[i], sells[i]
        price = X[i, 0]
        if short:
            price_bear = X_bear[i, 0]

            amount_usd_sell_bear = BUY * capital_coin_bear * price_bear * (1 - commissions * multiplier)
            amount_coin_sell_bear = capital_coin_bear * BUY

            capital_coin_bear -= amount_coin_sell_bear
            capital_usd += amount_usd_sell_bear

        amount_coin_buy = BUY * capital_usd / price * (1 - commissions * multiplier)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * price * (1 - commissions * multiplier)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        if short:
            amount_coin_buy_bear = SELL * capital_usd / price_bear * (1 - commissions * multiplier)
            amount_usd_buy_bear = capital_usd * SELL

            capital_coin_bear += amount_coin_buy_bear
            capital_usd -= amount_usd_buy_bear

        wealths[i] = capital_usd + capital_coin * price + capital_coin_bear * price_bear

    wealths = np.array(wealths) / wealths[0]

    return wealths

def get_wealths_limit(X, p, buys, sells = None, commissions = 0.00075):
    if sells is None:
        sells = 1 - buys
    capital_usd = 1000
    capital_coin = 0

    wealths = [0] * X.shape[0]

    can_buy = (X[1:, 2] <= X[:-1, 0] * (1 - p)).astype(float)
    buys[:-1] *= can_buy

    can_sell = (X[1:, 1] >= X[:-1, 0] * (1 + p)).astype(float)
    sells[:-1] *= can_sell

    for i in range(X.shape[0]):
        BUY, SELL = buys[i], sells[i]
        price = X[i, 0]

        amount_coin_buy = BUY * capital_usd / (price * (1 - p)) * (1 - commissions)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * (price * (1 + p)) * (1 - commissions)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        wealths[i] = capital_usd + capital_coin * price

    wealths = np.array(wealths) / wealths[0]

    return wealths

def get_wealths_oco(X, X_agg, aggregate_N, p, m, buys, sells = None, commissions = 0.00075, verbose = False):
    if sells is None:
        sells = 1 - buys
    capital_usd = 1000
    capital_coin = 0

    wealths = [0] * X_agg.shape[0]

    buy_lim_count = 0
    buy_stop_count = 0
    sell_lim_count = 0
    sell_stop_count = 0

    sideways_count = 0

    for i in range(X_agg.shape[0] - 1):
        BUY, SELL = buys[i], sells[i]
        price = X_agg[i, 0]
        buy_price = price
        sell_price = price

        if BUY > 0.0 and capital_usd > 0.0:
            idx = np.arange((i + 1) * aggregate_N, (i + 2) * aggregate_N)
            li_limit = X[idx, 2] <= price * np.exp(-m*p)
            li_stop = X[idx, 1] > price * (1 + p)

            if li_limit.any():
                first_limit_idx = idx[li_limit][0]
            else:
                first_limit_idx = np.inf

            if li_stop.any():
                first_stop_idx = idx[li_stop][0]
            else:
                first_stop_idx = np.inf

            if (first_limit_idx == first_stop_idx and first_limit_idx != np.inf) or \
                    (first_stop_idx < first_limit_idx):
                buy_price = price * (1 + p)
                buy_stop_count += 1
            elif first_limit_idx < first_stop_idx:
                buy_price = price * np.exp(-m*p)
                buy_lim_count += 1
            else:
                BUY = 0.0
                sideways_count += 1

        if SELL > 0.0 and capital_coin > 0.0:
            idx = np.arange((i + 1) * aggregate_N, (i + 2) * aggregate_N)
            li_limit = X[idx, 1] >= price * np.exp(m*p)
            li_stop = X[idx, 2] < price * (1 - p)

            if li_limit.any():
                first_limit_idx = idx[li_limit][0]
            else:
                first_limit_idx = np.inf

            if li_stop.any():
                first_stop_idx = idx[li_stop][0]
            else:
                first_stop_idx = np.inf

            if (first_limit_idx == first_stop_idx and first_limit_idx != np.inf) or \
                    (first_stop_idx < first_limit_idx):
                sell_price = price * (1 - p)
                sell_stop_count += 1
            elif first_limit_idx < first_stop_idx:
                sell_price = price * np.exp(m*p)
                sell_lim_count += 1
            else:
                SELL = 0.0
                sideways_count += 1

        amount_coin_buy = BUY * capital_usd / buy_price * (1 - commissions)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * sell_price * (1 - commissions)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        wealths[i] = capital_usd + capital_coin * price

    if verbose:
        buy_profit = np.exp((m*p * buy_lim_count - p * buy_stop_count) / (buy_lim_count + buy_stop_count))
        print(buy_lim_count, buy_stop_count, buy_profit)
        sell_profit = np.exp(m*p) * sell_lim_count + (1 - p) * sell_stop_count
        sell_profit /= sell_lim_count + sell_stop_count
        print(sell_lim_count, sell_stop_count, sell_profit)
        print(sideways_count)

    wealths[-1] = wealths[-2]

    wealths = np.array(wealths) / wealths[0]

    return wealths

def get_wealths_trailing_stop(X, X_agg, aggregate_N, p, buys, sells = None, commissions = 0.00075, verbose = False, short = False):
    if sells is None:
        sells = 1 - buys
    if p == 0.0:
        p = 1
    capital_usd = 1000
    capital_coin = 0
    capital_coin_bear = 0

    multiplier = 3

    price_bear = 0.0
    if short:
        X_bear = get_multiplied_X(X, -multiplier)
        X_agg_bear = aggregate(X_bear, aggregate_N)

    X = get_multiplied_X(X, multiplier)
    X_agg = aggregate(X, aggregate_N)

    wealths = [0] * X_agg.shape[0]

    stop_count = 0
    sideways_count = 0

    prev_BUY = 0.0
    prev_SELL = 0.0

    for i in range(X_agg.shape[0] - 1):
        BUY, SELL = buys[i], sells[i]
        buy_first = BUY > SELL
        price = X_agg[i, 0]
        buy_price = price
        sell_price = price
        if short:
            price_bear = X_agg_bear[i, 0]
            buy_price_bear = price_bear
            sell_price_bear = price_bear
        idx = np.arange((i + 1) * aggregate_N, (i + 2) * aggregate_N)

        if BUY == 1.0 and SELL == 0.0:
            if BUY > prev_BUY:
                stop_sell_price = price * (1 - p)
                if short:
                    stop_buy_price_bear = price_bear * (1 + p)
            stop_sell_price = max(stop_sell_price, price * (1 - p))
            if short:
                stop_buy_price_bear = min(stop_buy_price_bear, price_bear * (1 + p))
            li_stop = X[idx, 2] < stop_sell_price

            if li_stop.any():
                sell_price = stop_sell_price
                stop_buy_price = sell_price * (1 + p)
                if short:
                    buy_price_bear = stop_buy_price_bear
                    stop_sell_price_bear = buy_price_bear * (1 - p)
                SELL = 1.0
                stop_count += 1
            else:
                sideways_count += 1

        if SELL == 1.0 and BUY == 0.0:
            if SELL > prev_SELL:
                stop_buy_price = price * (1 + p)
                if short:
                    stop_sell_price_bear = price_bear * (1 - p)
            stop_buy_price = min(stop_buy_price, price * (1 + p))
            if short:
                stop_sell_price_bear = min(stop_sell_price_bear, price_bear * (1 - p))
            li_stop = X[idx, 1] > stop_buy_price

            if li_stop.any():
                buy_price = stop_buy_price
                stop_sell_price = buy_price * (1 - p)
                if short:
                    sell_price_bear = stop_sell_price_bear
                    stop_buy_price_bear = sell_price_bear * (1 + p)
                BUY = 1.0
                stop_count += 1
            else:
                sideways_count += 1

        # stop loss flips side
        if buy_first:
            if short:
                amount_usd_sell_bear = BUY * capital_coin_bear * sell_price_bear * (1 - commissions * multiplier)
                amount_coin_sell_bear = capital_coin_bear * BUY

                capital_coin_bear -= amount_coin_sell_bear
                capital_usd += amount_usd_sell_bear

            amount_coin_buy = BUY * capital_usd / buy_price * (1 - commissions * multiplier)
            amount_usd_buy = capital_usd * BUY

            amount_usd_sell = SELL * capital_coin * sell_price * (1 - commissions * multiplier)
            amount_coin_sell = capital_coin * SELL

            capital_coin += amount_coin_buy - amount_coin_sell
            capital_usd += amount_usd_sell - amount_usd_buy

            if short:
                amount_coin_buy_bear = SELL * capital_usd / buy_price_bear * (1 - commissions * multiplier)
                amount_usd_buy_bear = capital_usd * SELL

                capital_coin_bear += amount_coin_buy_bear
                capital_usd -= amount_usd_buy_bear
        else:
            amount_usd_sell = SELL * capital_coin * sell_price * (1 - commissions * multiplier)
            amount_coin_sell = capital_coin * SELL

            capital_coin -= amount_coin_sell
            capital_usd += amount_usd_sell

            if short:
                amount_coin_buy_bear = SELL * capital_usd / buy_price_bear * (1 - commissions * multiplier)
                amount_usd_buy_bear = capital_usd * SELL

                capital_coin_bear += amount_coin_buy_bear
                capital_usd -= amount_usd_buy_bear

                amount_usd_sell_bear = BUY * capital_coin_bear * sell_price_bear * (1 - commissions * multiplier)
                amount_coin_sell_bear = capital_coin_bear * BUY

                capital_coin_bear -= amount_coin_sell_bear
                capital_usd += amount_usd_sell_bear

            amount_coin_buy = BUY * capital_usd / buy_price * (1 - commissions * multiplier)
            amount_usd_buy = capital_usd * BUY

            capital_coin += amount_coin_buy
            capital_usd -= amount_usd_buy

        wealths[i] = capital_usd + capital_coin * price + capital_coin_bear * price_bear

        if BUY == 1.0 and SELL == 1.0:
            if buy_first:
                BUY = 0.0
            else:
                SELL = 0.0

        prev_BUY = BUY
        prev_SELL = SELL

    if verbose:
        print(stop_count / (stop_count + sideways_count))

    wealths[-1] = wealths[-2]

    wealths = np.array(wealths) / wealths[0]

    return wealths




def objective_function(buys, X, initial_usd, initial_coin, commissions):
    sells = 1 - buys
    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X, buys, sells, initial_usd, initial_coin, commissions
    )
    return -wealths[-1]

def filter_buys(X, buys, commissions):
    res = copy.deepcopy(np.round(buys))

    diffs = np.diff(np.concatenate([np.array([0]), res, np.array([0])]))
    idx = np.arange(buys.shape[0] + 1)
    buys_idx = idx[diffs == 1]
    sells_idx = idx[diffs == -1]

    for i in range(buys_idx.shape[0]):
        buy_i = min(buys_idx[i], X.shape[0] - 1)
        sell_i = min(sells_idx[i], X.shape[0] - 1)

        profit = X[sell_i, 0] / X[buy_i, 0] - 1 - commissions * 2

        if profit < 0:
            res[buy_i:sell_i] = 0

    diffs = np.diff(np.concatenate([np.array([0]), res, np.array([0])]))
    buys_idx = idx[diffs == 1]
    sells_idx = idx[diffs == -1]

    for i in range(buys_idx.shape[0] - 1):
        buy_i = min(buys_idx[i + 1], X.shape[0] - 1)
        sell_i = min(sells_idx[i], X.shape[0] - 1)

        lost_profit = X[buy_i, 0] / X[sell_i, 0] - 1 + commissions * 2

        if lost_profit > 0:
            res[sell_i:buy_i] = 1

    diffs = np.diff(np.concatenate([np.array([0]), res, np.array([0])]))
    buys_idx = idx[diffs == 1]
    sells_idx = idx[diffs == -1]

    for i in range(buys_idx.shape[0]):
        start_i = sells_idx[i - 1] if i > 0 else 0
        end_i = sells_idx[i] if i < sells_idx.shape[0] else X.shape[0] - 1

        min_i = np.argmin(X[start_i:end_i, 0]) + start_i

        res[start_i:min_i] = 0
        res[min_i:end_i] = 1

    diffs = np.diff(np.concatenate([np.array([0]), res, np.array([0])]))
    buys_idx = idx[diffs == 1]
    sells_idx = idx[diffs == -1]

    for i in range(sells_idx.shape[0]):
        start_i = buys_idx[i]
        end_i = buys_idx[i + 1] if i < buys_idx.shape[0] - 1 else X.shape[0] - 1

        max_i = np.argmax(X[start_i:end_i, 0]) + start_i

        res[start_i:max_i] = 1
        res[max_i:end_i] = 0

    return res

def get_optimal_strategy(coin_filenames, improve, batch_size, commissions, verbose, filename = None):
    X_all = load_all_data(coin_filenames)

    i_start = 0

    initial_capital = 1000
    capital_usd = initial_capital
    capital_coin = 0

    buys_out = []

    if filename is not None and os.path.exists(filename):
        df = pd.read_csv(filename, index_col = 0, header = None)
        if not improve:
            i_start = len(df)
            buys = df.values.reshape(-1)
            sells = 1 - buys
            buys_out.append(buys)

            _, capital_usd, capital_coin, _, _ = get_wealths(
                X_all[:len(df), :], buys, sells, capital_usd, capital_coin
            )

            print('Wealth: {}'.format((capital_usd + capital_coin * X_all[len(df) - 1, 0]) / initial_capital))
    else:
        improve = False

    for i in tqdm(range(i_start, X_all.shape[0], batch_size)):
        X = X_all[i:i+batch_size, :]
        bounds = Bounds(np.zeros(X.shape[0]), np.ones(X.shape[0]))

        if improve and i + batch_size <= len(df):
            x0 = df.values.reshape(-1)[i:i+batch_size]
        else:
            x0 = np.random.rand(X.shape[0])

        result = minimize(
            fun = objective_function,
            x0 = x0,
            args = (X, capital_usd, capital_coin, commissions),
            bounds = bounds
        )

        while result.success == False:
            result = minimize(
                fun = objective_function,
                x0 = result.x,
                args = (X, capital_usd, capital_coin, commissions),
                bounds = bounds
            )

        if verbose:
            print(-result.fun, X[-1, 0] / X[0, 0] - 1)

        buys = result.x
        sells = 1 - buys

        buys_out.append(buys)

        _, capital_usd, capital_coin, _, _ = get_wealths(
            X, buys, sells, capital_usd, capital_coin
        )


    buys_out = np.concatenate(buys_out)

    buys_out = filter_buys(X_all, buys_out, commissions)

    if verbose:
        wealths, _, _, _, _ = get_wealths(
            X_all, buys_out
        )
        print(wealths[-1], X_all[-1, 0] / X_all[0, 0])

    return X_all, buys_out

def save_optimum(coin, buys, dir):
    ser = pd.Series(buys)

    filename = dir + coin + '.csv'
    ser.to_csv(filename, header=False)

def save_optima(coin = None, improve = False, batch_size = 30, commissions = 0.00075, verbose = False):
    if coin is None:
        coin_dir_list = glob.glob('data/*/')
    else:
        coin_dir_list = glob.glob('data/' + coin + '/')

    dir = 'data/labels/'

    for coin_dirname in coin_dir_list:
        if coin_dirname != dir:
            coin = coin_dirname.split('/')[1]
            filename = dir + coin + '.csv'

            if not os.path.exists(dir):
                os.mkdir(dir)

            if not os.path.exists(filename):
                coin_filenames = glob.glob(coin_dirname + '*.json')
                X, buys = get_optimal_strategy(coin_filenames, improve, batch_size, commissions, verbose)
                save_optimum(coin, buys, dir)
                print(filename, 'processed')
            else:
                coin_filenames = glob.glob(coin_dirname + '*.json')
                X, buys = get_optimal_strategy(coin_filenames, improve, batch_size, commissions, verbose, filename)
                save_optimum(coin, buys, dir)
                print(filename, 'processed')



def visualize_optimum(X, buys):
    sells = 1 - buys
    wealths, _, _, _, _ = get_wealths(
        X, buys, sells
    )

    plt.style.use('seaborn')
    plt.plot(X[:, 0] / X[0, 0])
    plt.plot(wealths)
    plt.show()

if __name__ == '__main__':
    #coin = None
    coin = 'ETH'
    improve = False
    batch_size = 30 * 1
    commissions = 0.00125
    verbose = False

    start_time = time.time()
    save_optima(
        coin = coin,
        improve = improve,
        batch_size = batch_size,
        commissions = commissions,
        verbose = verbose
    )
    improve = True
    batch_size = 30 * 3
    save_optima(
        coin = coin,
        improve = improve,
        batch_size = batch_size,
        commissions = commissions,
        verbose = verbose
    )
    print('Time taken:', round_to_n(time.time() - start_time), 'seconds')
