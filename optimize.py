from scipy.optimize import minimize, LinearConstraint, Bounds
from data import load_data
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from utils import round_to_n, get_time
import os

# TODO: optimize training hyperparameters using bayesian optimization

def get_wealths(X, buys, sells = None, initial_usd = 1000, initial_coin = 0, commissions = 0.00075):
    if sells is None:
        sells = 1 - buys
    capital_usd = initial_usd
    capital_coin = initial_coin

    wealths = [0] * X.shape[0]
    buy_amounts = []
    sell_amounts = []

    for i in range(X.shape[0]):
        BUY, SELL = buys[i], sells[i]
        price = X[i, 0]

        amount_coin_buy = BUY * capital_usd / price * (1 - commissions)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * price * (1 - commissions)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        buy_amounts.append(amount_usd_buy)
        sell_amounts.append(amount_usd_sell)

        wealths[i] = capital_usd + capital_coin * price

    wealths = np.array(wealths) / wealths[0] - 1

    return wealths, capital_usd, capital_coin, buy_amounts, sell_amounts

def objective_function(buys, X, initial_usd, initial_coin):
    sells = 1 - buys
    wealths, capital_usd, capital_coin, buy_amounts, sell_amounts = get_wealths(
        X, buys, sells, initial_usd, initial_coin
    )
    return -wealths[-1]

def get_optimal_strategy(filename, sequence_length = 2001, batch_size = 30, verbose = False):
    X_all = load_data(filename, sequence_length, 0, 1)

    i = 0

    capital_usd = 1000
    capital_coin = 0

    buys_out = []

    while i < sequence_length:
        X = X_all[i:i+batch_size, :]
        bounds = Bounds(np.zeros(X.shape[0]), np.ones(X.shape[0]))
        x0 = np.random.rand(X.shape[0])

        result = minimize(
            fun = objective_function,
            x0 = x0,
            args = (X, capital_usd, capital_coin),
            bounds = bounds
        )

        while result.success == False:
            result = minimize(
                fun = objective_function,
                x0 = result.x,
                args = (X, capital_usd, capital_coin),
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

        i += batch_size

    buys_out = np.concatenate(buys_out)

    wealths, _, _, _, _ = get_wealths(
        X_all, buys_out
    )

    if verbose:
        print(wealths[-1], X_all[-1, 0] / X_all[0, 0] - 1)

    return X_all, buys_out

def save_optimum(filename, buys):
    ser = pd.Series(buys)

    t = str(get_time(filename))
    coin = filename.split('/')[1]

    dir = 'data/labels/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = 'data/labels/' + coin + '/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    if not os.path.exists(dir + t + '.csv'):
        ser.to_csv(dir + t + '.csv', header=False)

def save_optima(coin = None):
    if coin is None:
        file_list = glob.glob('data/*/*.json')
    else:
        file_list = glob.glob('data/' + coin + '/*.json')

    for filename in file_list:
        t = str(get_time(filename))
        coin = filename.split('/')[1]
        dir = 'data/labels/' + coin + '/'

        if not os.path.exists(dir + t + '.csv'):
            X, buys = get_optimal_strategy(filename)
            save_optimum(filename, buys)
            print(filename, 'processed')


def visualize_optimum(X, buys):
    sells = 1 - buys
    wealths, _, _, _, _ = get_wealths(
        X, buys, sells
    )

    plt.plot(range(X.shape[0]), X[:, 0] / X[0, 0])
    plt.plot(range(X.shape[0]), wealths + 1)
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    save_optima()
    print('Time taken:', round_to_n(time.time() - start_time), 'seconds')
