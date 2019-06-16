from model import build_model, build_model_old
import numpy as np
import glob
import matplotlib.pyplot as plt
import json
from data import load_data
from utils import get_time, calc_actions, calc_reward, calc_metrics
from sklearn.model_selection import train_test_split

# TODO: make these function parameters
latency = 0
commissions = 0.00075
initial_capital = 1000

def evaluate(filenames, weights_filename = None, dirname = None, input_length = 4 * 14):
    batch_size = 2001 - input_length - latency
    if dirname is not None:
        print(dirname)

    if weights_filename is None:
        weights_filenames = glob.glob('models/*.h5')
    else:
        weights_filenames = [weights_filename]

    for fname in weights_filenames:
        try:
            model = build_model()
            model.load_weights(fname)
        except ValueError:
            model = build_model_old()
            model.load_weights(fname)

        closes = np.zeros(shape=batch_size * len(filenames))
        wealths = np.zeros(shape=(batch_size + 2) * len(filenames))
        buy_amounts = []
        sell_amounts = []

        for i, filename in enumerate(filenames):
            X = load_data(filename, batch_size, input_length, latency)
            closes[i*batch_size:(i+1)*batch_size] = X[input_length:input_length+batch_size,0] / X[input_length,0]

            ws, bs, ss = calc_actions(model, X, batch_size, input_length, latency, initial_capital, commissions)
            wealths[i*(batch_size + 2):(i+1)*(batch_size + 2)] = ws + 1
            buy_amounts.extend(bs)
            sell_amounts.extend(ss)

            if i > 0:
                closes[i*batch_size:(i+1)*batch_size] *= closes[i*batch_size - 1]
                wealths[i*(batch_size + 2):(i+1)*(batch_size + 2)] *= wealths[i*(batch_size + 2) - 1]

        reward = calc_reward(wealths - 1)
        metrics = calc_metrics(reward, wealths - 1, buy_amounts, sell_amounts, initial_capital)

        print(fname)
        print(metrics)

        plt.plot(range(len(closes)), closes)
        plt.plot(range(len(wealths)), wealths)
        plt.show()


if __name__ == '__main__':
    for dir in glob.glob('data/*/'):
        files = glob.glob(dir + '*.json')
        print(len(files))
        evaluate(files, dirname = dir)
