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

    def buy(self, args):
        stoch = args['stoch']
        return stoch < self.buy_threshold

    def sell(self, args):
        stoch = args['stoch']
        return stoch > 1 - self.sell_threshold

class Heikin_ashi_criterion:
    def __init__(self):
        pass

    def buy(self, args):
        ha = args['ha']
        return ha[1] == ha[3]

    def sell(self, args):
        ha = args['ha']
        return ha[1] == ha[2]

class Trend_criterion:
    def __init__(self, threshold):
        self.threshold = threshold

    def buy(self, args):
        ma_corrected = args['ma_corrected']
        return ma_corrected > self.threshold

    def sell(self, args):
        ma_corrected = args['ma_corrected']
        return ma_corrected < -self.threshold

class Stop_loss_criterion:
    def __init__(self, stop_loss):
        self.stop_loss = stop_loss
        self.buy_price = None

    def buy(self, args):
        return self.buy_price is None

    def sell(self, args):
        close = args['close']
        return self.buy_price is not None and close / self.buy_price - 1 < self.stop_loss


class Take_profit_criterion:
    def __init__(self, take_profit):
        self.take_profit = take_profit
        self.buy_price = None

    def buy(self, args):
        return self.buy_price is None

    def sell(self, args):
        close = args['close']
        return self.buy_price is not None and close / self.buy_price - 1 > self.take_profit


class Deque_criterion:
    def __init__(self, maxlen, waiting_time):
        self.trades = deque(maxlen = maxlen)
        self.waiting_time = waiting_time
        self.sell_time = - self.waiting_time

    def get_profit(self):
        res = 1.0
        for trade in self.trades:
            res *= trade
        return res

    def append(self, trade):
        self.trades.append(trade)

    def get_waiting_time(self, profit):
        return self.waiting_time * (- profit * 100)

    def buy(self, args):
        current_time = args['current_time']
        profit = self.get_profit() - 1.0
        return profit >= 0.0 or current_time - self.sell_time > self.get_waiting_time(profit)

    def sell(self, args):
        return True


class Logic_criterion:
    def __init__(self):
        self.recently_bought = False
        self.recently_sold = False

    def buy(self, args):
        return not self.recently_bought

    def sell(self, args):
        return self.recently_bought and not self.recently_sold


class Base_Strategy:
    def __init__(self, params, stop_loss_take_profit, restrictive):
        self.stop_loss = Stop_loss_criterion(params['stop_loss'])
        self.take_profit = Take_profit_criterion(params['take_profit'])
        self.deque_criterion = Deque_criterion(params['maxlen'], params['waiting_time'])
        self.logic_criterion = Logic_criterion()

        self.buy_and_criteria = []

        self.sell_and_criteria = []

        self.buy_or_criteria = []

        self.sell_or_criteria = []

        if stop_loss_take_profit:
            self.buy_and_criteria.extend([
                self.stop_loss,
                self.take_profit,
            ])
            self.sell_or_criteria.extend([
                self.stop_loss,
                self.take_profit,
            ])

        if restrictive:
            self.buy_and_criteria.extend([
                self.deque_criterion,
                self.logic_criterion
            ])
            self.sell_and_criteria.extend([
                self.logic_criterion,
                self.deque_criterion
            ])


    def get_buy_and_rules(self, args):
        return all(map(lambda x: x.buy(args), self.buy_and_criteria))

    def get_sell_and_rules(self, args):
        return all(map(lambda x: x.sell(args), self.sell_and_criteria))

    def get_buy_or_rules(self, args):
        return any(map(lambda x: x.buy(args), self.buy_or_criteria))

    def get_sell_or_rules(self, args):
        return any(map(lambda x: x.sell(args), self.sell_or_criteria))

    def buy(self, args):
        return self.get_buy_and_rules(args) and self.get_buy_or_rules(args)

    def sell(self, args):
        return self.get_sell_and_rules(args) and self.get_sell_or_rules(args)

    def update_after_buy(self, price):
        self.take_profit.buy_price = price
        self.stop_loss.buy_price = price
        self.logic_criterion.recently_bought = True
        self.logic_criterion.recently_sold = False

    def update_after_sell(self, price, i):
        if self.take_profit.buy_price is not None:
            trade = price / self.take_profit.buy_price
            self.deque_criterion.append(trade)
        self.take_profit.buy_price = None
        self.stop_loss.buy_price = None
        self.deque_criterion.sell_time = i
        self.logic_criterion.recently_bought = False
        self.logic_criterion.recently_sold = True

    def reset(self):
        self.take_profit.buy_price = None
        self.stop_loss.buy_price = None
        self.deque_criterion.sell_time = -1
        self.deque_criterion.trades.clear()
        self.logic_criterion.recently_bought = False
        self.logic_criterion.recently_sold = False


class Simple_Strategy(Base_Strategy):
    def __init__(self, params, stop_loss_take_profit = False, restrictive = False):
        super().__init__(params, stop_loss_take_profit, restrictive)
        self.window_size = params['window_size1'] * 14

        self.stochastic_criterion = Stochastic_criterion(params['buy_threshold'], params['sell_threshold'])

        self.buy_or_criteria.append(
            self.stochastic_criterion
        )

        self.sell_or_criteria.append(
            self.stochastic_criterion
        )


    def get_output(self, X, reset = True):
        stochastic = stochastic_oscillator(X, self.window_size)

        sequence_length = stochastic.shape[0]

        X = X[-sequence_length:, :]

        buys = np.zeros(sequence_length, dtype=bool)
        sells = np.zeros(sequence_length, dtype=bool)

        for i in range(sequence_length):
            price = X[i, 0]
            stoch = stochastic[i]

            args = {
                'close': price,
                'stoch': stoch,
                'current_time': i,
            }

            if self.buy(args):
                self.update_after_buy(price)
                buys[i] = True
            elif self.sell(args):
                self.update_after_sell(price, i)
                sells[i] = True

        if reset:
            self.reset()

        return buys, sells


class Strategy(Base_Strategy):
    def __init__(self, params, stop_loss_take_profit = False, restrictive = False):
        super().__init__(params, stop_loss_take_profit, restrictive)
        self.window_size1 = params['window_size1'] * 14
        self.window_size2 = params['window_size2'] * 14
        self.window_size3 = params['window_size3'] * 14

        self.stochastic_criterion = Stochastic_criterion(params['buy_threshold'], params['sell_threshold'])
        self.ha_criterion = Heikin_ashi_criterion()
        self.trend_criterion = Trend_criterion(params['trend_threshold'])

        self.buy_and_criteria.extend([
            self.ha_criterion
        ])

        self.sell_and_criteria.extend([
            self.ha_criterion
        ])

        self.buy_or_criteria.extend([
            self.stochastic_criterion,
            self.trend_criterion
        ])

        self.sell_or_criteria.extend([
            self.stochastic_criterion,
            self.trend_criterion
        ])


    def get_output(self, X, reset = True):
        ha = heikin_ashi(X)

        tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
        ma = sma(tp, self.window_size1)
        sd = std(tp, self.window_size1)

        X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)

        stochastic = stochastic_oscillator(X_corrected, self.window_size2)

        X_corrected /= np.repeat(X[-X_corrected.shape[0]:, 0].reshape((-1, 1)), 4, axis = 1)

        ma_corrected = sma(X_corrected, self.window_size3)

        sequence_length = np.min([X_corrected.shape[0], ma_corrected.shape[0], stochastic.shape[0], ma.shape[0]])

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
        rolling_min_width = pd.Series(width).rolling(self.window_size2).min().dropna().values

        buys = np.zeros(sequence_length, dtype=bool)
        sells = np.zeros(sequence_length, dtype=bool)

        for i in range(sequence_length):
            price = X[i, 0]
            stoch = stochastic[i]

            args = {
                'close': price,
                'stoch': stoch,
                'current_time': i,
                'ma_corrected': ma_corrected[i],
                'ha': ha[i, :]
            }

            if self.buy(args):
                self.update_after_buy(price)
                buys[i] = True
            elif self.sell(args):
                self.update_after_sell(price, i)
                sells[i] = True

        if reset:
            self.reset()

        return buys, sells




def evaluate_strategy(files, strategy):
    X = load_all_data(files)

    buys, sells = strategy.get_output(X)
    sequence_length = buys.shape[0]
    X = X[-sequence_length:, :]

    initial_capital = 1000
    commissions = 0.00075

    capital_usd = initial_capital
    capital_coin = 0

    wealths = []
    trades = []

    buy_price = None

    i = 0
    while i < sequence_length:
        price = X[i, 0]

        if buys[i]:
            buy_price = price

            amount_coin = capital_usd / price * (1 - commissions)
            capital_coin += amount_coin
            capital_usd = 0
        elif sells[i]:
            if buy_price is not None:
                trade = price / buy_price
                trades.append(trade - 1)
            buy_price = None

            amount_usd = capital_coin * price * (1 - commissions)
            capital_usd += amount_usd
            capital_coin = 0

        wealths.append(capital_usd + capital_coin * price)

        i += 1

    wealths = np.array(wealths) / wealths[0]

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
    stop_loss_take_profit = True
    restrictive = True

    params = {
        'stop_loss': -0.03,
        'take_profit': 0.01,
        'maxlen': 3,
        'waiting_time': 8 * 60,
        'buy_threshold': 0.04,
        'sell_threshold': 0.07,
        'trend_threshold': 0.02,
        'window_size1': 3,
        'window_size2': 1,
        'window_size3': 1,
    }

    strategy = Strategy(params, stop_loss_take_profit, restrictive)

    dir = 'data/ETH/'
    # for dir in glob.glob('data/*/'):
    test_files = glob.glob(dir + '*.json')
    test_files.sort(key = get_time)
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files, strategy)
    print()
    test_files = test_files[-20:]
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files, strategy)
    print()
    test_files = test_files[-3:]
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    evaluate_strategy(test_files, strategy)
