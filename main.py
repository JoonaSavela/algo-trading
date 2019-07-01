import matplotlib.pyplot as plt
from utils import plot_y, stochastic_oscillator, heikin_ashi, sma
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
import plotly.plotly as py
import plotly.graph_objs as go

def main():

    times = []
    capitals = []
    bnbs = []

    for file in glob.glob('trading_logs/*.json'):
        with open(file, 'r') as f:
            obj = json.load(f)
            times.append(int(obj['initial_time']))
            times.append(int(obj['final_time']))
            capitals.append(float(obj['initial_capital']))
            capitals.append(float(obj['final_capital']))
            bnbs.append(float(obj['initial_bnb']))
            bnbs.append(float(obj['final_bnb']))

    times = np.array(times)
    capitals = np.array(capitals)
    bnbs = np.array(bnbs)

    i = np.argsort(times)
    times = times[i]
    capitals = capitals[i]
    bnbs = bnbs[i]

    times = (times - times[0]) / 60
    capitals = capitals / capitals[0]
    bnbs = bnbs / bnbs[0]

    plt.plot(times, capitals)
    plt.plot(times, bnbs)
    plt.show()

    # for file in glob.glob('data/*/*.json'):
    #     X = load_data(file, 2001, 0, 0)
    #     zero_count = np.sum(X == 0, axis = 0)
    #     if np.any(zero_count > 0):
    #         print(file, zero_count)

    # test_files = glob.glob('data/*/*.json')[14:15]
    # window_size1 = 1 * 14
    # window_size2 = window_size1
    # k = 1
    # # window_size = np.max([window_size1 + k - 1, window_size2])
    # window_size = window_size1 + window_size2
    # latency = 0
    # # sequence_length = 2001 - window_size + 2 - latency - k + 1
    # sequence_length = 4*60
    # print(sequence_length)
    #
    # for file in test_files:
    #     X = load_data(file, sequence_length, latency, window_size + 1, k)
    #     # print(np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1)))
    #
    #     print(X.shape)
    #     # ha = heikin_ashi(X)
    #     # print(ha[-sequence_length:, :].shape)
    #
    #     ma = sma(X, window_size2)
    #     print(ma.shape)
    #
    #     stochastic1 = stochastic_oscillator(X, window_size1, k)
    #     print(stochastic1.shape)
    #
    #     X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)
    #
    #     stochastic2 = stochastic_oscillator(X_corrected, window_size1, k)
    #     print(stochastic2.shape)
    #
    #     fig, axes = plt.subplots(figsize=(12, 8), ncols=2, nrows=2)
    #
    #     axes[0][0].plot(range(sequence_length), X[-sequence_length:, :4])
    #     axes[0][0].plot(range(sequence_length), ma[-sequence_length:])
    #     axes[1][0].plot(range(sequence_length), stochastic1[-sequence_length:])
    #     axes[0][1].plot(range(sequence_length), X_corrected[-sequence_length:])
    #     axes[1][1].plot(range(sequence_length), stochastic2[-sequence_length:])
    #     plt.show()

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
