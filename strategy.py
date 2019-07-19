from data import load_data, load_all_data
from utils import stochastic_oscillator, heikin_ashi, sma, std, get_time, round_to_n
import glob
from collections import deque
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)

import numpy as np
import pandas as pd

class Stochastic_criterion:
    def __init__(self, buy_threshold, sell_threshold = None):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold if sell_threshold is not None else buy_threshold

    def buy(self, stoch):
        return stoch < self.buy_threshold

    def sell(self, stoch):
        return stoch > 1 - self.sell_threshold

class Heikin_ashi_criterion:
    def __init__(self):
        pass

    def buy(self, ha):
        return ha[1] == ha[3]

    def sell(self, ha):
        return ha[1] == ha[2]

class Trend_criterion:
    def __init__(self, threshold):
        self.threshold = threshold

    def buy(self, ma_corrected):
        return ma_corrected > self.threshold

    def sell(self, ma_corrected):
        return ma_corrected < -self.threshold

class Stop_loss_criterion:
    def __init__(self, stop_loss):
        self.stop_loss = stop_loss
        self.buy_price = None

    def buy(self):
        return self.buy_price is None

    def sell(self, close):
        return self.buy_price is not None and close / self.buy_price - 1 < self.stop_loss


class Take_profit_criterion:
    def __init__(self, take_profit):
        self.take_profit = take_profit
        self.buy_price = None

    def buy(self):
        return self.buy_price is None

    def sell(self, close):
        return self.buy_price is not None and close / self.buy_price - 1 > self.take_profit


class Deque_criterion:
    def __init__(self, maxlen, waiting_time):
        self.trades = deque(maxlen = maxlen)
        self.waiting_time = waiting_time
        self.sell_time = - self.waiting_time
        self.recently_bought = False
        self.recently_sold = False

    def get_profit(self):
        res = 1.0
        for trade in self.trades:
            res *= trade
        return res

    def append(self, trade):
        self.trades.append(trade)

    def get_waiting_time(self, profit):
        return self.waiting_time * (- profit * 100)

    def buy(self, current_time):
        profit = self.get_profit() - 1.0
        return profit >= 0.0 or current_time - self.sell_time > self.get_waiting_time(profit)

    def sell(self):
        return True


class Logic_criterion:
    def __init__(self):
        self.recently_bought = False
        self.recently_sold = False

    def buy(self):
        return not self.recently_bought

    def sell(self):
        return self.recently_bought and not self.recently_sold


def evaluate_strategy(files):
    window_size1 = 3 * 14
    window_size2 = 1 * 14
    window_size3 = 1 * 14
    k = 1
    latency = 0

    initial_capital = 1000
    commissions = 0.00075

    stochastic_criterion = Stochastic_criterion(0.04, 0.07)
    ha_criterion = Heikin_ashi_criterion()
    stop_loss = Stop_loss_criterion(-0.02)
    take_profit = Take_profit_criterion(0.01)
    trend_criterion = Trend_criterion(0.02)
    deque_criterion = Deque_criterion(3, 12 * 60)
    logic_criterion = Logic_criterion()

    trades = []
    current_time = 0

    X = load_all_data(files)

    # stochastic = stochastic_oscillator(X, window_size, k)
    ha = heikin_ashi(X)

    tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
    ma = sma(tp, window_size1)
    sd = std(tp, window_size1)

    X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)

    stochastic = stochastic_oscillator(X_corrected, window_size2, k, latency)

    X_corrected /= np.repeat(X[-X_corrected.shape[0]:, 0].reshape((-1, 1)), 4, axis = 1)

    ma_corrected = sma(X_corrected, window_size3)

    sequence_length = np.min([X_corrected.shape[0], ma_corrected.shape[0], stochastic.shape[0], ma.shape[0]])
    # print(sequence_length)

    X = X[-sequence_length:, :]
    ha = ha[-sequence_length:, :]
    ma = ma[-sequence_length:]
    stochastic = stochastic[-sequence_length:]
    ma_corrected = ma_corrected[-sequence_length:]
    sd = sd[-sequence_length:]

    m = 2 # TODO: move
    upper = ma + m * sd
    lower = ma - m * sd
    width = (upper - lower) / ma
    rolling_min_width = pd.Series(width).rolling(window_size2).min().dropna().values

    capital_usd = initial_capital
    trade_start_capital = capital_usd
    capital_coin = 0

    wealths = [initial_capital]
    capital_usds = [capital_usd]
    capital_coins = [capital_coin]
    buy_amounts = []
    sell_amounts = []

    i = 0
    while i < sequence_length:
        price = X[i, 0]
        stoch = stochastic[i]

        if ha_criterion.buy(ha[i, :]) and take_profit.buy() and \
                stop_loss.buy() and deque_criterion.buy(current_time) and \
                logic_criterion.buy() and \
                (stochastic_criterion.buy(stoch) or \
                trend_criterion.buy(ma_corrected[i])):
            take_profit.buy_price = price
            stop_loss.buy_price = price
            logic_criterion.recently_bought = True
            logic_criterion.recently_sold = False

            amount_coin = capital_usd / price * (1 - commissions)
            capital_coin += amount_coin
            capital_usd = 0
            buy_amounts.append(amount_coin * price)
        elif ha_criterion.sell(ha[i, :]) and logic_criterion.sell() and \
                (stochastic_criterion.sell(stoch) or \
                trend_criterion.sell(ma_corrected[i]) or \
                stop_loss.sell(price) or \
                take_profit.sell(price)):
            if take_profit.buy_price is not None:
                trade = price / take_profit.buy_price
                trades.append(trade - 1)
                deque_criterion.append(trade)
            take_profit.buy_price = None
            stop_loss.buy_price = None
            deque_criterion.sell_time = current_time
            logic_criterion.recently_bought = False
            logic_criterion.recently_sold = True

            amount_usd = capital_coin * price * (1 - commissions)
            capital_usd += amount_usd
            capital_coin = 0
            sell_amounts.append(amount_usd)

        wealths.append(capital_usd + capital_coin * price)
        capital_usds.append(capital_usd)
        capital_coins.append(capital_coin * price)

        i += 1
        current_time += 1

    price = X[-1, 0]
    amount_usd = capital_coin * price * (1 - commissions)
    capital_usd += amount_usd
    sell_amounts.append(amount_usd)
    capital_coin = 0

    wealths.append(capital_usd)
    capital_usds.append(capital_usd)
    capital_coins.append(capital_coin * price)

    wealths = np.array(wealths) / wealths[0]
    capital_usds = np.array(capital_usds) / initial_capital
    capital_coins = np.array(capital_coins) / initial_capital

    closes = X[:,0] / X[0, 0]

    print('profit:', wealths[-1])
    print('benchmark profit:', closes[-1])
    print('min, max:', np.min(wealths), np.max(wealths))

    plt.plot(range(closes.shape[0]), closes)
    plt.plot(range(wealths.shape[0]), wealths)
    plt.show()

    print(len(trades), np.mean(trades), np.min(trades), np.max(trades))
    plt.hist(trades)
    plt.show()

if __name__ == '__main__':
    dir = 'data/ETH/'
    # for dir in glob.glob('data/*/'):
    test_files = glob.glob(dir + '*.json')
    test_files.sort(key = get_time)
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files)
    print()
    test_files = test_files[-20:]
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files)
    print()
    test_files = test_files[-3:]
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files)
