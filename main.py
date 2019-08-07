import matplotlib.pyplot as plt
import matplotlib as mpl
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
from utils import get_time, round_to_n
import pymc3 as pm
from pymc3.distributions import Interpolated
import theano
from scipy import stats
import logging
# logger = logging.getLogger("pymc3")
# logger.setLevel(logging.ERROR)

def from_posterior(param, samples):
    smin, smax = np.min(samples), np.max(samples)
    width = smax - smin
    x = np.linspace(smin, smax, 100)
    y = stats.gaussian_kde(samples)(x)

    # what was never sampled should have a small probability but not 0,
    # so we'll extend the domain and use linear approximation of density on it
    x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
    y = np.concatenate([[0], y, [0]])
    return Interpolated(param, x, y)

def main():
    plt.style.use('seaborn')

    # for file in glob.glob('data/*/*.json'):
    #     X = load_data(file, 2001, 0, 0)
    #     zero_count = np.sum(X == 0, axis = 0)
    #     if np.any(zero_count > 0):
    #         print(file, zero_count)

    test_files = glob.glob('data/ETH/*.json')
    test_files.sort(key = get_time)

    # idx = np.array([2, 4, 5, -1])
    # idx = np.array([0, 1, 2, 3])
    idx = np.arange(1)
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

    # tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
    # c = 70
    # lips = sma(tp, 5 * c)
    # teeth = sma(tp, 8 * c)
    # jaws = sma(tp, 13 * c)
    #
    # sequence_length = jaws.shape[0]
    #
    # X = X[-sequence_length:, :]
    # lips = lips[-sequence_length:]
    # teeth = teeth[-sequence_length:]
    # jaws = jaws[-sequence_length:]
    #
    # up = (lips > teeth) & (teeth > jaws)
    # down = (lips < teeth) & (teeth < jaws)
    # neutral = ~(up | down)
    #
    # t = np.arange(sequence_length)
    #
    # plt.plot(t[up], X[up, 0], 'g.')
    # plt.plot(t[down], X[down, 0], 'r.')
    # plt.plot(t[neutral], X[neutral, 0], 'b.')
    # plt.plot(np.arange(sequence_length), lips[-sequence_length:], 'g')
    # plt.plot(np.arange(sequence_length) + 0 * c, teeth[-sequence_length:], 'r')
    # plt.plot(np.arange(sequence_length) + 0 * c, jaws[-sequence_length:], 'b')
    # plt.show()

    returns = X[1:, 0] / X[:-1, 0] - 1
    print(np.std(returns))
    traces = []
    quantiles = []
    stds = []

    n_sample = 200
    cores = 2
    sigma = 0.004
    print(int(n_sample * cores))

    start_time = time.time()

    mu0 = theano.shared(0.0, name='hyper_mu0')
    sd0 = theano.shared(0.01, name='hyper_sd0')
    output = theano.shared(returns[:1])

    with pm.Model() as model:
        mu = pm.StudentT('mu', mu=mu0, sd=sd0, nu=1, testval=0.)

        r = pm.StudentT('return', mu=mu, sd=sigma, nu=1, observed=output)

    for i in range(120):
        if i > 0:
            mu0.set_value(trace['mu'].mean())
            sd0.set_value(trace['mu'].std())

        output.set_value(returns[i:i+1])

        with model:
            # approx = pm.fit()
            # trace = approx.sample(n_sample)
            trace = pm.sample(n_sample, progressbar=True, cores=cores, n_init=200_000)
            traces.append(trace)

        q = np.quantile(trace['mu'], (0.1, 0.5, 0.9))
        sd = np.std(trace['mu'])

        string = '{}: true return: {}, quantiles: {}, std: {}'.format(
            i,
            returns[i],
            q,
            sd
        )
        print(string)

        quantiles.append(q)
        stds.append(sd)

        # pm.traceplot(trace)
        # plt.show()
        #
        # # plt.hist(returns, bins=30, alpha=0.7, density=True)
        # plt.hist(trace['mu'], bins=30, alpha=0.7, density=True)
        # xs = [0, returns[i]]
        # for x in xs:
        #     plt.axvline(x, c='k')
        # plt.show()

    print('Time taken:', round_to_n(time.time() - start_time), 'seconds')
    print(traces[0]['mu'].shape)

    cmap = mpl.cm.plasma
    for update_i, trace in enumerate(traces):
        samples = trace['mu']
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)
        plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
    # for i in range(len(traces)):
    #     plt.axvline(returns[i], c=cmap(1 - i / len(traces)))
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    quantiles = np.stack(quantiles)

    plt.plot(returns[:len(traces)], label='returns')
    plt.plot(quantiles[:, 1], label='median')
    plt.fill_between(range(len(quantiles)), quantiles[:, 0], quantiles[:, 2], alpha=0.3, label='CI')
    plt.legend()
    plt.show()

    plt.plot(stds)
    plt.show()

    # plt.plot(X[1:, 0] / X[:-1, 0])
    # plt.show()


if __name__ == "__main__":
    main()
