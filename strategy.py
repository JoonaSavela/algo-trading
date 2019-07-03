from data import load_data
from utils import stochastic_oscillator, heikin_ashi, sma, std
import glob
import matplotlib.pyplot as plt
import numpy as np

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

class Bollinger_criterion:
    def __init__(self, m):
        self.m = m

    def buy(self, close, ma, std):
        return close < ma - self.m * std

    def sell(self, close, ma, std):
        return close > ma + self.m * std

class Stop_loss_criterion:
    def __init__(self, stop_loss):
        self.stop_loss = stop_loss

    def buy(self):
        return True

    def sell(self, close, open):
        return close / open - 1 < self.stop_loss


class Take_profit_criterion:
    def __init__(self, take_profit):
        self.take_profit = take_profit
        self.buy_price = None

    def buy(self):
        return self.buy_price is None

    def sell(self, close):
        return self.buy_price is not None and close / self.buy_price - 1 > self.take_profit



def calc_actions():
    pass

def evaluate_strategy(files):
    window_size1 = 3 * 14
    window_size2 = 1 * 14
    k = 1
    latency = 0
    sequence_length = 2001 - window_size1 - window_size2 + 2 - latency - k + 1
    # print(sequence_length)

    initial_capital = 1000
    commissions = 0.00075

    stochastic_criterion = Stochastic_criterion(0.04)
    ha_criterion = Heikin_ashi_criterion()
    # bollinger_criterion = Bollinger_criterion(8) # Useless?
    stop_loss = Stop_loss_criterion(-0.0075)
    take_profit = Take_profit_criterion(0.01)

    cs = np.zeros(shape=sequence_length * len(files))
    ws = np.zeros(shape=(sequence_length + 2) * len(files))

    for file_i, file in enumerate(files):
        X = load_data(file, sequence_length, latency, window_size1 + window_size2 - 1, k)

        # stochastic = stochastic_oscillator(X, window_size, k)
        ha = heikin_ashi(X)

        tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
        ma = sma(tp, window_size1)
        # stds = std(tp, window_size)

        X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)
        stochastic = stochastic_oscillator(X_corrected, window_size2, k, latency)
        # print(stochastic.shape)

        X = X[-sequence_length:, :]
        ha = ha[-sequence_length:, :]
        ma = ma[-sequence_length:]
        stochastic = stochastic[-sequence_length:]
        # stds = stds[-sequence_length:]

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

            if ha_criterion.buy(ha[i, :]) and stochastic_criterion.buy(stoch) and \
                    take_profit.buy():
                take_profit.buy_price = price
                amount_coin = capital_usd / price * (1 - commissions)
                capital_coin += amount_coin
                capital_usd = 0
                buy_amounts.append(amount_coin * price)
            elif ha_criterion.sell(ha[i, :]) and \
                    (stochastic_criterion.sell(stoch) or \
                    stop_loss.sell(price, X[i, 3]) or \
                    take_profit.sell(price)):
                take_profit.buy_price = None
                amount_usd = capital_coin * price * (1 - commissions)
                capital_usd += amount_usd
                capital_coin = 0
                sell_amounts.append(amount_usd)

            wealths.append(capital_usd + capital_coin * price)
            capital_usds.append(capital_usd)
            capital_coins.append(capital_coin * price)

            i += 1

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

        cs[file_i*sequence_length:(file_i+1)*sequence_length] = X[:,0] / X[0, 0]
        ws[file_i*(sequence_length + 2):(file_i+1)*(sequence_length + 2)] = wealths

        if file_i > 0:
            cs[file_i*sequence_length:(file_i+1)*sequence_length] *= cs[file_i*sequence_length - 1]
            ws[file_i*(sequence_length + 2):(file_i+1)*(sequence_length + 2)] *= ws[file_i*(sequence_length + 2) - 1]

    print('profit:', ws[-1])
    print('benchmark profit:', cs[-1])
    print('min, max:', np.min(ws), np.max(ws))

    plt.plot(range(cs.shape[0]), cs)
    plt.plot(range(ws.shape[0]), ws)
    plt.show()

if __name__ == '__main__':
    # print(len(glob.glob('data/*/*.json')))
    # for dir in glob.glob('data/*/'):
    dir = 'data/ETH/'
    test_files = glob.glob(dir + '*.json')[:-1]
    # print(dir, len(test_files))
    evaluate_strategy(test_files)
