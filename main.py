import matplotlib.pyplot as plt
from utils import plot_y
from model import build_model
from data import get_and_save_all
import numpy as np
import json
import requests
import glob
import pandas as pd
import time

def main():
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
