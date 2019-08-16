# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from model import RelationalMemory, FFN
from data import load_data
from utils import calc_actions, calc_reward, calc_metrics
from utils import *
from sklearn.model_selection import train_test_split
from evaluate import evaluate
import copy
import torch
import torch.nn as nn
from optimize import get_wealths


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

    dir = 'data/ETH/'
    files = glob.glob(dir + '*.json')
    files.sort(key = get_time)

    train(
        files = files,
        model = model,
        n_epochs = n_epochs,
        lr = lr,
    )
