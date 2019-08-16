# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from model import *
from data import load_all_data
from utils import *
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from optimize import get_wealths

def plot_labels(files, coin, n = 200):
    X = load_all_data(files)

    df = pd.read_csv(
        'data/labels/' + coin + '.csv',
        index_col = 0,
        header = None,
        nrows = n,
    )

    buys_optim = df.values.reshape(-1)

    wealths, _, _, _, _ = get_wealths(
        X[:n, :], buys_optim
    )

    print(wealths[-1])

    plt.style.use('seaborn')
    plt.plot(buys_optim, label='buys')
    plt.plot((X[:n, 0] / X[0, 0] - 1) * 100, c='k', alpha=0.5, label='price')
    plt.plot(wealths * 100, label='wealth')
    plt.legend()
    plt.show()

def train(files, model, n_epochs, lr):
    X = load_all_data(files)

    for e in range(n_epochs):
        break


if __name__ == '__main__':
    commissions = 0.00075

    inputs = {
        'close',

    }

    input_size = len(inputs)
    lr = 0.001

    # NN model definition
    model = FFN(input_size)

    n_epochs = 20
    print_step = 1#max(n_epochs // 20, 1)

    coin = 'ETH'
    dir = 'data/{}/'.format(coin)
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    train(
        files = files,
        model = model,
        n_epochs = n_epochs,
        lr = lr,
    )

    plot_labels(files, coin)
