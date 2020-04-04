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
from optimize import *
from peaks import get_peaks
from tqdm import tqdm
from keys import ftx_api_key, ftx_secret_key
from parameter_search import *
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from itertools import product
from ftx.rest.client import FtxClient
from trade import *

# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: include running quantile in sells

# TODO: try only shorting?

# TODO: plot different statistics of a strategy, use them to find improvements

# TODO: find different ways of optimizing parameters?

def main():
    trading_pipeline()

    # client = FtxClient(ftx_api_key, ftx_secret_key)
    # total, res = get_total_balance(client, True)
    # print(res)
    # print()
    # sell_assets(client, 'ETHBEAR')
    # buy_assets_short(client, 1, res['usdValue']['ETHBEAR'] / total, 0.001)
    # time.sleep(1)
    # _, res = get_total_balance(client, True)
    # print(res)
    # print()
    # sell_assets_short(client, 1, 3, res['usdValue']['ETH'] / total, 0.001)
    # time.sleep(1)
    # _, res = get_total_balance(client, True)
    # print(res)

    # balances = pd.DataFrame(client.get_balances())#.set_index('coin')
    # balances = balances.set_index('coin')['usdValue']
    # print(balances)
    # print(balances.get('ETH', 0))
    # markets = pd.DataFrame(client.list_markets())
    # res = markets[(markets['baseCurrency'] == 'ETHBEAR') & (markets['quoteCurrency'] == 'USDT')]
    # print(res)
    # print(res['ask'])


    # plot_displacement([
    #                     (4, 12, 1, 3, 0),
    #                     (3, 16, 1, 3, 0), # best
    #                   ])

    # aggregate_N_list = range(1, 13)
    # w_list = range(1, 51)
    # m_list = range(1, 4, 2)
    # m_bear_list = [3]
    # p_list = np.arange(0, 0.9, 0.1)
    # # p_list = [0]
    #
    # print(len(list(product(aggregate_N_list, w_list, m_list, m_bear_list, p_list))))
    #
    # # aggregate_N, w, m, m_bear, p
    # params = find_optimal_aggregated_strategy(
    #             aggregate_N_list,
    #             w_list,
    #             m_list,
    #             m_bear_list,
    #             p_list,
    #             N_repeat = 1,
    #             verbose = True,
    #             disable = False,
    #             randomize = True,
    #             short = True,
    #             trailing_stop=True)

    # plot_performance([
    #                     (4, 12, 1, 3, 0),
    #                     (3, 16, 1, 3, 0), # best
    #                   ],
    #                   N_repeat = 1000,
    #                   short = True,
    #                   trailing_stop = True)





if __name__ == "__main__":
    main()
