import matplotlib.pyplot as plt
from utils import plot_y, stochastic_oscillator
from model import RelationalMemory
from data import get_and_save_all, load_data
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy

def main():

    # test_files = glob.glob('data/*/*.json')[:1]
    # sequence_length = 4*60
    # window_size = 3 * 14
    # latency = 0
    #
    # for file in test_files:
    #     fig, axes = plt.subplots(figsize=(18, 8), ncols=3, nrows=2)
    #     for i, k in enumerate(range(3, 8, 2)):
    #         X = load_data(file, sequence_length, latency, window_size, k)
    #         print(X.shape)
    #         stoch = stochastic_oscillator(X, window_size, k)
    #
    #         print(stoch.shape)
    #
    #         axes[0][i].plot(range(stoch.shape[0]), X[window_size - 1 + k - 1:, 0] / X[window_size - 1 + k - 1, 0])
    #         axes[1][i].plot(range(stoch.shape[0]), stoch)
    #
    #     plt.show()


    # test_files = [glob.glob('data/*/*.json')[0]]
    # sequence_length = 4*60
    # window_size = 3 * 14
    # latency = 0
    #
    # for file in test_files:
    #     X = load_data(file, sequence_length, latency, window_size)
    #     print(X.shape)
    #     stoch = stochastic_oscillator(X, window_size)
    #
    #     print(stoch.shape)
    #
    #     fig, axes = plt.subplots(figsize=(6, 8), ncols=1, nrows=2)
    #
    #     axes[0].plot(range(stoch.shape[0]), X[window_size - 1:, 0] / X[window_size - 1, 0])
    #     axes[1].plot(range(stoch.shape[0]), stoch)
    #     plt.show()

    for filename in glob.glob('run_summaries/*.csv'):
        print(filename)
        table = pd.read_csv(filename)
        n = table.shape[0]
        print(n)

        print(np.prod(table['profit'] + 1))
        print(np.power(np.prod(table['profit'] + 1), 1/n))
        print(np.prod(table['max_profit'] + 1))
        print(np.power(np.prod(table['max_profit'] + 1), 1/n))
        print(np.prod(table['min_profit'] + 1))
        print(np.power(np.prod(table['min_profit'] + 1), 1/n))

        plt.plot(table['iteration'], table['profit'])
        plt.plot(table['iteration'], table['max_profit'])
        plt.plot(table['iteration'], table['min_profit'])
        plt.show()

        plt.plot(table['iteration'], table['reward'])
        plt.show()

    # for filename in glob.glob('data/*/*.json'):
    #     with open(filename, 'r') as file:
    #         obj = json.load(file)
    #     data = obj['Data']
    #
    #     X = np.zeros(shape=(len(data), 6))
    #     for i in range(len(data)):
    #         item = data[i]
    #         tmp = []
    #         for key, value in item.items():
    #             if key != 'time':
    #                 tmp.append(value)
    #         X[i, :] = tmp
    #
    #     plt.plot(range(len(data)), X[:, 0] / X[0, 0])
    #     plt.title(filename)
    #     plt.show()


if __name__ == "__main__":
    main()
