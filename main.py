import matplotlib.pyplot as plt
from utils import stochastic_oscillator, heikin_ashi, sma, std
from model import RelationalMemory
from data import get_and_save_all, load_data, load_all_data
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy
import plotly.plotly as py
import plotly.graph_objs as go
from utils import get_time

def main():

    # for file in glob.glob('data/*/*.json'):
    #     X = load_data(file, 2001, 0, 0)
    #     zero_count = np.sum(X == 0, axis = 0)
    #     if np.any(zero_count > 0):
    #         print(file, zero_count)

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([2, 4, 5, -1])
    idx = np.array([0, 1])
    test_files = np.array(test_files)[idx]
    window_size1 = 14#3 * 14
    window_size2 = 4 * 60#1 * 14
    window_size3 = 1 * 14
    k = 1
    # window_size = np.max([window_size1 + k - 1, window_size2])
    window_size = window_size1 + window_size2 - 1
    latency = 0
    sequence_length = 2001 - window_size + 1 - latency - k + 1
    # sequence_length = 8*60
    # print(sequence_length)

    # for file in test_files:
    # X = load_data(file, sequence_length, latency, window_size, k)
    X = load_all_data(test_files)
    # print(np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1)))

    # print(X.shape)
    # ha = heikin_ashi(X)
    # print(ha[-sequence_length:, :].shape)

    tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
    c = 20
    lips = sma(tp, 5 * c)
    teeth = sma(tp, 8 * c)
    jaws = sma(tp, 13 * c)

    sequence_length = jaws.shape[0]

    plt.plot(range(sequence_length), X[-sequence_length:, 0])
    plt.plot(np.arange(sequence_length), lips[-sequence_length:], 'g')
    plt.plot(np.arange(sequence_length) + 0 * c, teeth[-sequence_length:], 'r')
    plt.plot(np.arange(sequence_length) + 0 * c, jaws[-sequence_length:], 'b')
    plt.show()

    # trace = go.Candlestick(x=list(range(X.shape[0])),
    #         open=X[:, 3],
    #         high=X[:, 1],
    #         low=X[:, 2],
    #         close=X[:, 0])
    # data = [trace]
    # py.plot(data, filename='simple_candlestick')
    #
    # trace1 = go.Candlestick(x=list(range(X.shape[0] - 1)),
    #         open=ha[:, 1],
    #         high=ha[:, 2],
    #         low=ha[:, 3],
    #         close=ha[:, 0])
    # data1 = [trace1]
    # py.plot(data1, filename='heikin-ashi_candlestick')



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

    # for filename in glob.glob('run_summaries/*.csv'):
    #     print(filename)
    #     table = pd.read_csv(filename)
    #     n = table.shape[0]
    #     print(n)
    #
    #     print(np.prod(table['profit'] + 1))
    #     print(np.power(np.prod(table['profit'] + 1), 1/n))
    #     print(np.prod(table['max_profit'] + 1))
    #     print(np.power(np.prod(table['max_profit'] + 1), 1/n))
    #     print(np.prod(table['min_profit'] + 1))
    #     print(np.power(np.prod(table['min_profit'] + 1), 1/n))
    #
    #     plt.plot(table['iteration'], table['profit'])
    #     plt.plot(table['iteration'], table['max_profit'])
    #     plt.plot(table['iteration'], table['min_profit'])
    #     plt.show()
    #
    #     plt.plot(table['iteration'], table['reward'])
    #     plt.show()

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
