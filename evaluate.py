from model import build_model, build_model_old
import numpy as np
import glob
import matplotlib.pyplot as plt
import json
from data import load_data
from utils import get_time, calc_actions, calc_reward, calc_metrics

input_length = 4 * 14
latency = 1
batch_size = 2000 - input_length - latency
commissions = 0.00075

def evaluate():
    model = build_model()
    files = glob.glob('data/ETH/*.json')
    filename = max(files, key=get_time)
    X = load_data(filename, batch_size, input_length, latency)
    price_max = np.max(X[input_length:input_length+batch_size,0]) / X[input_length, 0]
    price_min = np.min(X[input_length:input_length+batch_size,0]) / X[input_length, 0]

    for fname in glob.glob('models/*.h5'):
        try:
            model.load_weights(fname)
        except ValueError:
            model = build_model_old()
            model.load_weights(fname)

        initial_capital = 1000

        wealths, buy_amounts, sell_amounts = calc_actions(model, X, batch_size, input_length, latency, initial_capital, commissions)

        reward = calc_reward(wealths)

        metrics = calc_metrics(reward, wealths, buy_amounts, sell_amounts, initial_capital)

        print(fname)
        print(metrics)

        plt.plot(range(batch_size), X[input_length:input_length+batch_size,0] / X[input_length, 0])
        plt.plot(range(len(wealths)), wealths + 1)
        plt.show()

        # plt.plot(range(len(wealths)), wealths)
        # plt.show()

        # plt.plot(range(len(capital_usds)), capital_usds)
        # plt.plot(range(len(capital_coins)), np.array(capital_coins) * X[input_length,0])
        # plt.show()


if __name__ == '__main__':
    evaluate()
