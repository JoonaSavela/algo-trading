import matplotlib.pyplot as plt
import matplotlib as mpl
from data import *
import numpy as np
import pandas as pd
import time
from utils import *
from model import *
from optimize import *
from tqdm import tqdm
from parameter_search import *
from itertools import product

# TODO: train a (bayesian) NN on the (aggregated) data

# TODO: include running quantile in sells?

# TODO: plot different statistics of a strategy, use them to find improvements

# TODO: find different ways of optimizing parameters?



def main():

    # take_profit_long, take_profit_short = \
    # get_take_profits([
    #                     # (4, 12, 1, 3),
    #                     (3, 16, 1, 3), # best
    #                   ],
    #                   short = True,
    #                   N_repeat = 500,
    #                   randomize = True,
    #                   commissions = 0.00075,
    #                   step = 0.01,
    #                   verbose = True)

    # plot_displacement([
    #                     (4, 12, 1, 3, 1.19, 2.14),
    #                     (3, 16, 1, 3, 1.19, 2.14), # best
    #                   ])

    # plot_performance([
    #                     (3, 16, 1, 3, 1.19, 2.14), # best
    #                     # (4, 12, 1, 3, 1.19, 2.14),
    #                   ],
    #                   N_repeat = 1,
    #                   short = True,
    #                   take_profit = True)





if __name__ == "__main__":
    main()
