from data import load_data
from utils import stochastic_oscillator
import glob
import matplotlib.pyplot as plt
import numpy as np

def calc_actions():
    pass

def evaluate_strategy(files):
    window_size = 3 * 14
    latency = 0
    k = 7
    sequence_length = 2001 - window_size + 1 - latency - k + 1

    initial_capital = 1000
    commissions = 0.00075
    wait_time = 20
    stoch_threshold = 0.2
    stop_loss = -0.0025

    cs = np.zeros(shape=sequence_length * len(files))
    ws = np.zeros(shape=(sequence_length + 2) * len(files))

    for file_i, file in enumerate(files):
        X = load_data(file, sequence_length, latency, window_size, k)
        stochastic = stochastic_oscillator(X, window_size, k)
        X = X[window_size - 1 + k - 1:, :]

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
            price = X[i + latency, 0]
            stoch = stochastic[i]

            if stoch < stoch_threshold:
                amount_coin = capital_usd / price * (1 - commissions)
                capital_coin += amount_coin
                capital_usd = 0
                buy_amounts.append(amount_coin * price)
            elif stoch > 1 - stoch_threshold:
                amount_usd = capital_coin * price * (1 - commissions)
                capital_usd += amount_usd
                capital_coin = 0
                sell_amounts.append(amount_usd)

            wealths.append(capital_usd + capital_coin * price)
            capital_usds.append(capital_usd)
            capital_coins.append(capital_coin * price)

            # profit = wealths[-1] / trade_start_capital - 1
            # if profit < stop_loss:
            #     amount_usd = capital_coin * price * (1 - commissions)
            #     capital_usd += amount_usd
            #     sell_amounts.append(amount_usd)
            #     capital_coin = 0
            #     trade_start_capital = capital_usd
            #     j = 0
            #     while j < wait_time - 1 and i < sequence_length - 1:
            #         wealths.append(capital_usd + capital_coin * price)
            #         capital_usds.append(capital_usd)
            #         capital_coins.append(capital_coin * price)
            #         i += 1
            #         j += 1

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

        # print(wealths.shape)
        # print(sequence_length)

        cs[file_i*sequence_length:(file_i+1)*sequence_length] = X[:,0] / X[0, 0]
        ws[file_i*(sequence_length + 2):(file_i+1)*(sequence_length + 2)] = wealths

        if file_i > 0:
            cs[file_i*sequence_length:(file_i+1)*sequence_length] *= cs[file_i*sequence_length - 1]
            ws[file_i*(sequence_length + 2):(file_i+1)*(sequence_length + 2)] *= ws[file_i*(sequence_length + 2) - 1]

    print('profit:', ws[-1])
    print('benchmark profit:', cs[-1])

    plt.plot(range(cs.shape[0]), cs)
    plt.plot(range(ws.shape[0]), ws)
    plt.show()

if __name__ == '__main__':
    test_files = glob.glob('data/*/*.json')[:6]
    evaluate_strategy(test_files)
