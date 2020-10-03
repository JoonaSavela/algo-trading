from scipy.optimize import minimize, LinearConstraint, Bounds
from data import load_all_data
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from utils import *
import os
from tqdm import tqdm
import copy

# TODO: combine all versions into a single function?
def get_wealths(X, buys, sells = None, commissions = 0.0007, spread_bull = 0.0001, X_bear = None, spread_bear = 0.001):
    short = X_bear is not None
    if sells is None:
        sells = 1 - buys
    capital_usd = 1000
    capital_coin = 0
    capital_coin_bear = 0

    price_bear = 0.0

    wealths = [0] * X.shape[0]

    for i in range(X.shape[0]):
        BUY, SELL = buys[i], sells[i]
        price = X[i, 0]
        if short:
            price_bear = X_bear[i, 0]

            amount_usd_sell_bear = BUY * capital_coin_bear * price_bear * (1 - commissions - spread_bear)
            amount_coin_sell_bear = capital_coin_bear * BUY

            capital_coin_bear -= amount_coin_sell_bear
            capital_usd += amount_usd_sell_bear

        amount_coin_buy = BUY * capital_usd / price * (1 - commissions - spread_bull)
        amount_usd_buy = capital_usd * BUY

        amount_usd_sell = SELL * capital_coin * price * (1 - commissions - spread_bull)
        amount_coin_sell = capital_coin * SELL

        capital_coin += amount_coin_buy - amount_coin_sell
        capital_usd += amount_usd_sell - amount_usd_buy

        if short:
            amount_coin_buy_bear = SELL * capital_usd / price_bear * (1 - commissions - spread_bear)
            amount_usd_buy_bear = capital_usd * SELL

            capital_coin_bear += amount_coin_buy_bear
            capital_usd -= amount_usd_buy_bear

        wealths[i] = capital_usd + capital_coin * price + capital_coin_bear * price_bear

    wealths = np.array(wealths) / wealths[0]

    return wealths


# TODO: remove trade_wealths from input, add output?
def append_trade_wealths(wealths, entries, exits, N, trade_wealths, max_trade_wealths):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_wealths = wealths[entry_i:exit_i]
        trade_wealths.append(sub_wealths[-1] / sub_wealths[0])
        max_trade_wealths.append(sub_wealths.max() / sub_wealths[0])

def append_sub_trade_wealths(X, entries, exits, N, sub_trade_wealths, min_sub_trade_wealths):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_X = X[entry_i:exit_i, :]
        sub_wealths = sub_X[1:, 0] / sub_X[:-1, 0]
        min_sub_wealths = sub_X[1:, 2] / sub_X[:-1, 0]
        sub_trade_wealths.append(sub_wealths)
        min_sub_trade_wealths.append(min_sub_wealths)


def get_take_profit_wealths_from_trades(trade_wealths, max_trade_wealths, take_profit, total_months, commissions, spread, return_total = True):
    trade_wealths = np.array(trade_wealths)
    max_trade_wealths = np.array(max_trade_wealths)
    trade_wealths[max_trade_wealths > take_profit] = take_profit * (1 - commissions - spread)

    if return_total:
        return np.exp(np.sum(np.log(trade_wealths)) / total_months)

    return trade_wealths

def get_stop_loss_wealths_from_sub_trades(sub_trade_wealths, min_sub_trade_wealths, stop_loss, total_months, commissions, spread, return_total = True):
    trade_wealths = []
    for i in range(len(sub_trade_wealths)):
        sub_trade_wealths_i = np.copy(sub_trade_wealths[i])
        li = min_sub_trade_wealths[i] < stop_loss
        sub_trade_wealths_i[li] = stop_loss * (1 - commissions - spread) ** 2
        trade_wealths.append(np.prod(sub_trade_wealths_i) * (1 - commissions - spread) ** 2)

    trade_wealths = np.array(trade_wealths)
    if return_total:
        return np.exp(np.sum(np.log(trade_wealths)) / total_months)

    return trade_wealths

def transform_wealths(wealths, X, entries, exits, N, take_profit, stop_loss, commissions, spread):
    assert(take_profit == np.Inf or stop_loss == 0)
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N

        # Process take_profit
        if take_profit != np.Inf:
            sub_wealths = wealths[entry_i:exit_i]
            li = sub_wealths / sub_wealths[0] > take_profit
            if np.any(li):
                first_i = np.arange(entry_i, exit_i)[li][0]
                old_wealth = wealths[exit_i - 1]
                wealths[first_i:exit_i] = wealths[entry_i] * take_profit * (1 - commissions - spread)
                wealths[exit_i:] *= wealths[exit_i - 1] / old_wealth

        # Process stop_loss
        if stop_loss != 0:
            sub_X = X[entry_i:exit_i, :]
            li = sub_X[1:, 2] / sub_X[:-1, 0] < stop_loss
            for i in np.arange(entry_i + 1, exit_i)[li]:
                old_wealth = wealths[i]
                wealths[i:] *= wealths[i - 1] * stop_loss * (1 - commissions - spread) ** 2 / old_wealth



if __name__ == '__main__':
    pass
