import matplotlib.pyplot as plt
import matplotlib as mpl
from data import get_and_save_all, load_data, load_all_data
import numpy as np
import json
import requests
import glob
import pandas as pd
import time
import torch
import copy
from utils import *
from model import *
from optimize import get_wealths
from peaks import get_peaks

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)
    # print('Number of months:', round_to_n(len(test_files) * 2000 / (60 * 24 * 30), 3))

    # idx = np.array([0, 1, 2, 3])
    # idx = np.arange(2)
    test_files = np.array(test_files)[:1]
    X = load_all_data(test_files)

    # alpha0 = 0.85
    x = 12
    alpha0 = (x - 1) / x
    # print(alpha0, 1 / (1 - alpha0))
    # print((x - 1) / x)
    x = 26
    alpha1 = (x - 1) / x
    # alpha1 = 0.9975
    n = 2

    logit0 = p_to_logit(alpha0)
    logit1 = p_to_logit(alpha1)

    if n > 1:
        alphas = logit_to_p(np.linspace(logit0, logit1, n))
    else:
        alphas = [alpha0]
    # print(alphas)

    emas = []
    for i in range(n):
        emas.append(ema(X[:, 0] / X[0, 0], alphas[i], 1.0))

    macd = emas[0] - emas[1]
    # macd = ema(macd, 0.25)

    buys = macd > 0
    sells = macd <= 0


    buys = buys.astype(float)
    sells = sells.astype(float)

    wealths, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0.00025
    )
    wealths += 1

    wealths1, _, _, _, _ = get_wealths(
        X, buys, sells, commissions = 0
    )
    wealths1 += 1

    print(wealths[-1], wealths1[-1])
    plt.plot(X[:, 0] / X[0, 0], c='k', alpha=0.5)
    for i in range(len(emas)):
        plt.plot(emas[i], c='b', alpha=0.65)
    plt.plot(wealths, c='g')
    plt.plot(wealths1, c='g', alpha = 0.5)
    if np.log(wealths1[-1]) / np.log(10) > 2:
        plt.yscale('log')
    plt.show()





if __name__ == "__main__":
    main()
