import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
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
from optimize import get_wealths, get_wealths_limit
from peaks import get_peaks
from tqdm import tqdm
import binance
from binance.client import Client
from binance.enums import *
from keys import binance_api_key, binance_secret_key

# TODO: simulate OCO

def main():
    plt.style.use('seaborn')

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    X = load_all_data(test_files, 0)

    aggregate_N = 60 * 11
    w = 4

    X = aggregate(X, aggregate_N)

    high_diff = X[1:, 1] / X[:-1, 0] - 1
    print(np.median(high_diff))
    N = high_diff.shape[0]
    N2 = np.round(1 * N).astype(int)

    high_diff = np.sort(high_diff)[:N2]

    low_diff = X[1:, 2] / X[:-1, 0] - 1
    print(np.median(low_diff))

    low_diff = np.sort(low_diff)[:N2]

    print(np.mean(np.min(high_diff) == high_diff))
    print(np.mean(np.max(low_diff) == low_diff))

    plt.hist(high_diff, 50)
    plt.show()

    plt.hist(low_diff, 50)
    plt.show()




if __name__ == "__main__":
    main()
