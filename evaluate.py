from model import RelationalMemory
import numpy as np
import glob
import matplotlib.pyplot as plt
import json
from data import load_data
from utils import get_time, calc_actions, calc_reward, calc_metrics
from sklearn.model_selection import train_test_split
import torch

# TODO: make these function parameters
latency = 0
commissions = 0.00075
initial_capital = 1000

def evaluate(filenames, state_dict_filename = None, dirname = None, input_size = 2, seq_length = 1, mem_slots = 8, num_heads = 2, window_size = 3 * 14):
    sequence_length = 2001 - window_size  + 1 - latency
    if dirname is not None:
        print(dirname)

    if state_dict_filename is None:
        state_dict_filenames = glob.glob('models/*.pt')
    else:
        state_dict_filenames = [state_dict_filename]

    # NN model definition
    model = RelationalMemory(mem_slots=mem_slots, head_size=input_size, input_size=input_size, num_heads=num_heads, num_blocks=1, forget_bias=1., input_bias=0.)

    for fname in state_dict_filenames:
        model.load_state_dict(torch.load(fname))

        closes = np.zeros(shape=sequence_length * len(filenames))
        wealths = np.zeros(shape=(sequence_length + 2) * len(filenames))
        buy_amounts = []
        sell_amounts = []

        for i, filename in enumerate(filenames):
            X = load_data(filename, sequence_length, latency, window_size)
            closes[i*sequence_length:(i+1)*sequence_length] = X[window_size - 1:window_size - 1 + sequence_length,0] / X[window_size - 1, 0]

            ws, bs, ss, capital_usds, capital_coins = calc_actions(model, X, sequence_length, latency, window_size, initial_capital, commissions)
            wealths[i*(sequence_length + 2):(i+1)*(sequence_length + 2)] = ws + 1
            buy_amounts.extend(bs)
            sell_amounts.extend(ss)

            if i > 0:
                closes[i*sequence_length:(i+1)*sequence_length] *= closes[i*sequence_length - 1]
                wealths[i*(sequence_length + 2):(i+1)*(sequence_length + 2)] *= wealths[i*(sequence_length + 2) - 1]

        # TODO: fix this
        # reward = calc_reward(wealths - 1)
        reward = np.mean(wealths - 1)
        metrics = calc_metrics(reward, wealths - 1, buy_amounts, sell_amounts, initial_capital, closes[0], closes[-1])

        print(fname)
        print(metrics)

        plt.plot(range(len(closes)), closes)
        plt.plot(range(len(wealths)), wealths)
        plt.show()


if __name__ == '__main__':
    test_files = glob.glob('data/*/*.json')[:1]
    evaluate(test_files, 'models/model_state_dict.pt', None)
    # evaluate(test_files, 'models/model_state_dict_1561200009.051622.pt', None)
    # for dir in glob.glob('data/*/'):
    #     files = glob.glob(dir + '*.json')
    #     print(len(files))
    #     evaluate(files, dirname = dir)
