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
from parameters import parameters

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
    def __init__(self, threshold = None):
        self.threshold = threshold if threshold is not None else 0.0

    def buy(self, args):
        ha = args['ha']
        return ha[1] / ha[3] - 1 >= self.threshold

    def sell(self, args):
        ha = args['ha']
        return 1 - ha[1] / ha[2] <= self.threshold

class Stop_loss_criterion:
    def __init__(self, stop_loss, decay):
        self.stop_loss = stop_loss
        self.decay = decay
        self.buy_price = None
        self.buy_time = None

    def buy(self, args):
        return self.buy_price is None and self.buy_time is None

    def sell(self, args):
        if self.buy_time is None or self.buy_price is None:
            return False
        close = args['close']
        current_time = args['current_time']
        decayed_stop_loss = (self.stop_loss + 1) * (self.decay + 1) ** (current_time - self.buy_time) - 1
        return close / self.buy_price - 1 < decayed_stop_loss


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


class Bollinger_squeeze_criterion:
    def __init__(self, min_threshold, change_threshold):
        self.min_threshold = min_threshold
        self.change_threshold = change_threshold

    def buy(self, args):
        rolling_min_width = args['rolling_min_width']
        mean_width = args['mean_width']
        width = args['width']
        return mean_width / rolling_min_width - 1 < self.min_threshold and \
            width / mean_width - 1 > self.change_threshold

    def sell(self, args):
        return self.buy(args)


class Alligator_criterion:
    def __init__(self):
        pass

    def sell(self, args):
        lips = args['lips']
        teeth = args['teeth']
        jaws = args['jaws']
        return lips < teeth and teeth < jaws

    def buy(self, args):
        return not self.sell(args)



class Base_Strategy:
    def __init__(self, params, stop_loss_take_profit, restrictive):
        self.stop_loss = Stop_loss_criterion(params['stop_loss'] if stop_loss_take_profit else -1, params['decay'] if stop_loss_take_profit else 0)
        self.take_profit = Take_profit_criterion(params['take_profit'] if stop_loss_take_profit else 1)
        self.deque_criterion = Deque_criterion(
            int(params['maxlen']) if restrictive else 3,
            (int(params['waiting_time']) if restrictive else 1)
        )
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

    def update_after_buy(self, price, current_time):
        self.take_profit.buy_price = price
        self.stop_loss.buy_price = price
        self.stop_loss.buy_time = current_time
        self.logic_criterion.recently_bought = True
        self.logic_criterion.recently_sold = False

    def update_after_sell(self, price, current_time):
        if self.take_profit.buy_price is not None:
            trade = price / self.take_profit.buy_price
            self.deque_criterion.append(trade)
        self.take_profit.buy_price = None
        self.stop_loss.buy_price = None
        self.stop_loss.buy_time = None
        self.deque_criterion.sell_time = current_time
        self.logic_criterion.recently_bought = False
        self.logic_criterion.recently_sold = True

    def reset(self):
        self.take_profit.buy_price = None
        self.stop_loss.buy_price = None
        self.deque_criterion.sell_time = -1
        self.deque_criterion.trades.clear()
        self.logic_criterion.recently_bought = False
        self.logic_criterion.recently_sold = False

    def size(self):
        return 1

    @staticmethod
    def get_options(stop_loss_take_profit, restrictive):
        options = {}

        options['name'] = 'Base_Stratey'

        if stop_loss_take_profit:
            options['stop_loss'] = ('uniform', (-0.2, 0.0))
            options['decay'] = ('uniform', (0.0, 0.001))
            options['take_profit'] = ('uniform', (0.0, 0.2))

        if restrictive:
            options['maxlen'] = ('range', (2, 10))
            options['waiting_time'] = ('range', (0, 24 * 60))

        return options


class Main_Strategy(Base_Strategy):
    def __init__(self, params, stop_loss_take_profit = False, restrictive = False):
        super().__init__(params, stop_loss_take_profit, restrictive)
        self.window_size1 = int(params['window_size1'])
        self.window_size2 = int(params['window_size2'])

        self.stochastic_criterion = Stochastic_criterion(params['buy_threshold'], params['sell_threshold'])
        self.ha_criterion = Heikin_ashi_criterion(params['ha_threshold'])

        self.window_size = int(params['window_size'])
        self.look_back_size = int(params['look_back_size'])
        self.rolling_min_window_size = int(params['rolling_min_window_size'])

        self.bollinger_squeeze_criterion = Bollinger_squeeze_criterion(
            params['min_threshold'],
            params['change_threshold']
        )

        self.c = int(params['c'])

        self.alligator_criterion = Alligator_criterion()

        self.buy_and_criteria.extend([
            self.ha_criterion,
            self.alligator_criterion
        ])

        self.sell_and_criteria.extend([
            self.ha_criterion
        ])

        self.buy_or_criteria.extend([
            self.stochastic_criterion,
            self.bollinger_squeeze_criterion
        ])

        self.sell_or_criteria.extend([
            self.stochastic_criterion,
            self.bollinger_squeeze_criterion,
            self.alligator_criterion
        ])


    def get_output(self, X, current_time, reset = True, update = True):
        ha = heikin_ashi(X)

        tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
        ma = sma(tp, self.window_size1)
        lips = sma(tp, 5 * self.c)
        teeth = sma(tp, 8 * self.c)
        jaws = sma(tp, 13 * self.c)

        X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)

        stochastic = stochastic_oscillator(X_corrected, self.window_size2)

        ma = sma(tp, self.window_size)
        sd = std(tp, self.window_size)

        upper = ma + sd
        lower = ma - sd
        width = (upper - lower) / ma
        rolling_min_width = pd.Series(width).rolling(self.rolling_min_window_size).min().dropna().values

        width_sma = sma(width.reshape((width.shape[0], 1)), self.look_back_size)

        sequence_length = np.min([rolling_min_width.shape[0], width_sma.shape[0], X_corrected.shape[0], stochastic.shape[0], ma.shape[0], jaws.shape[0]])

        X = X[-sequence_length:, :]
        ha = ha[-sequence_length:, :]
        ma = ma[-sequence_length:]
        stochastic = stochastic[-sequence_length:]
        lips = lips[-sequence_length:]
        teeth = teeth[-sequence_length:]
        jaws = jaws[-sequence_length:]

        buys = np.zeros(sequence_length - 1, dtype=bool)
        sells = np.zeros(sequence_length - 1, dtype=bool)

        sequence_length -= 1

        for i in range(sequence_length):
            price = X[i + 1, 0]
            stoch = stochastic[i + 1]

            args = {
                'close': price,
                'stoch': stoch,
                'current_time': current_time + i,
                'ha': ha[i + 1, :],
                'rolling_min_width': rolling_min_width[i],
                'mean_width': width_sma[i],
                'width': width[i + 1],
                'lips': lips[i + 1],
                'teeth': teeth[i + 1],
                'jaws': jaws[i + 1],
            }

            if self.buy(args):
                if update:
                    self.update_after_buy(price, current_time + i)
                buys[i] = True
            elif self.sell(args):
                if update:
                    self.update_after_sell(price, current_time + i)
                sells[i] = True

        if reset:
            self.reset()

        if buys.shape[0] == 1:
            return buys[0], sells[0], \
                list(map(lambda x: x.buy(args), self.buy_and_criteria)), \
                list(map(lambda x: x.sell(args), self.sell_and_criteria)), \
                list(map(lambda x: x.buy(args), self.buy_or_criteria)), \
                list(map(lambda x: x.sell(args), self.sell_or_criteria))

        return buys, sells

    def size(self):
        return np.max([self.window_size + np.max([self.rolling_min_window_size, self.look_back_size]),
            self.window_size1 + self.window_size2,
            13 * self.c + 1
        ])


    @staticmethod
    def get_options(stop_loss_take_profit, restrictive):
        options = super(Main_Strategy, Main_Strategy).get_options(stop_loss_take_profit, restrictive)

        options['name'] = 'Main_Strategy'

        options['window_size1'] = ('range', (3, 5 * 14 * 14))
        options['window_size2'] = ('range', (3, 5 * 14 * 14))

        options['window_size'] = ('range', (5, 5 * 14 * 14))
        options['look_back_size'] = ('range', (3, 5 * 14 * 14))
        options['rolling_min_window_size'] = ('range', (3, 5 * 14 * 14))

        options['min_threshold'] = ('uniform', (0.0, 4.0))
        options['change_threshold'] = ('uniform', (0.0, 4.0))

        options['buy_threshold'] = ('uniform', (0.0, 0.3))
        options['sell_threshold'] = ('uniform', (0.0, 0.3))

        options['c'] = ('range', (1, 150))

        options['ha_threshold'] = ('uniform', (0.0, 0.0005))

        return options



def evaluate_strategy(X, strategy, start = 0, current_time = 0, verbose = True):
    X = X[start:, :]

    n_months = X.shape[0] / (60 * 24 * 30)

    if verbose:
        print('Number of months:', n_months)

    buys, sells = strategy.get_output(X, current_time)
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

    trades = np.array(trades)

    losing_trades = (trades < 0).astype(int)
    idx = np.arange(1, len(trades))

    begins = idx[np.diff(losing_trades) == 1]
    ends = idx[np.diff(losing_trades) == -1]

    losses = []

    for i in range(min([len(ends), len(begins)])):
        loss = np.prod(trades[begins[i]:ends[i]] + 1)
        losses.append(loss)

    try:
        biggest_loss = min(losses)
    except ValueError as e:
        biggest_loss = 1.0


    if verbose:
        print('profit:', wealths[-1], wealths[-1] ** (1 / n_months))
        print('benchmark profit:', closes[-1])
        print('min, max:', np.min(wealths), np.max(wealths))
        print('biggest loss:', biggest_loss)

        plt.plot(range(closes.shape[0]), closes)
        plt.plot(range(wealths.shape[0]), wealths)
        plt.show()

        print(len(trades), np.std(trades), np.mean(trades), np.min(trades), np.max(trades))
        plt.hist(trades)
        plt.show()

    # wealths -= 1

    return wealths[-1], np.min(wealths), np.max(wealths), trades, biggest_loss



if __name__ == '__main__':
    stop_loss_take_profit = True
    restrictive = True

    obj = parameters[2]

    params = obj['params']

    for k, v in Main_Strategy.get_options(stop_loss_take_profit, restrictive).items():
        if k != 'name' and k not in params:
            type, bounds = v
            params[k] = bounds[0]

    # params['take_profit'] = 0.02
    # params['waiting_time'] = 60 * 12
    # params['maxlen'] = 3

    print(params)

    strategy = Main_Strategy(params, stop_loss_take_profit, restrictive)

    dir = 'data/ETH/'
    test_files = glob.glob(dir + '*.json')

    test_files.sort(key = get_time)
    print(dir, len(test_files), round_to_n(len(test_files) * 2001 / (60 * 24)))
    X = load_all_data(test_files)
    starts = [0]#, X.shape[0] // 2, X.shape[0] * 3 // 4]

    for start in starts:
        evaluate_strategy(X, strategy, start)
        print()
