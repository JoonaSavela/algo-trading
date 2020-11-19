from scipy.optimize import minimize, LinearConstraint, Bounds
from data import load_all_data, get_average_spread
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
from utils import *
import os
from tqdm import tqdm
import copy
from trade import get_total_balance

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

# TODO: handle situation: short == False
# TODO: can this be made faster?
def get_wealths_fast(X, buys, sells = None, commissions = 0.0007, spread_bull = 0.0001, X_bear = None, spread_bear = 0.001):
    short = X_bear is not None
    if sells is None:
        sells = 1 - buys
    N = buys.shape[0]
    buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)

    wealths = np.ones((N,))

    if buys_idx[0] < sells_idx[0]:
        buy_state = True
        entry_i = buys_idx[0]
    else:
        buy_state = False
        entry_i = sells_idx[0]

    while entry_i < N - 1:
        if buy_state:
            larger_exits_idx = sells_idx[sells_idx > entry_i]
        else:
            larger_exits_idx = buys_idx[buys_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        # print(entry_i, exit_i, buy_state)

        if buy_state:
            sub_X = X[entry_i:exit_i, 0] / X[entry_i, 0]
            commissions_and_spread = (1 - commissions - spread_bull)
        else:
            sub_X = X_bear[entry_i:exit_i, 0] / X_bear[entry_i, 0]
            commissions_and_spread = (1 - commissions - spread_bear)

        old_wealth = wealths[entry_i]

        wealths[entry_i:exit_i] = sub_X * old_wealth * commissions_and_spread
        wealths[exit_i - 1] *= commissions_and_spread

        entry_i = exit_i - 1
        buy_state = not buy_state

    # print(wealths)

    return wealths

# TODO: add comments
# TODO: or split this into multiple functions
def get_adaptive_wealths(X, buys_dict, sells_dict, strategy, total_balance, potential_balances, potential_spreads, commissions = 0.0007, X_bear = None):
    short = X_bear is not None
    key_spreads_float = [float(spread) for spread in strategy.keys()]

    # TODO: move this into a separate function?
    spread = potential_spreads[np.argmin(np.abs(potential_balances - total_balance))]
    commissions_and_spread = (1 - commissions - spread)
    key_spread = str(key_spreads_float[np.argmin(np.abs(key_spreads_float - spread))])
    aggregate_N, w, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = strategy[key_spread]['params']

    buys_idx_dict = {}
    sells_idx_dict = {}

    N = buys_dict[key_spread].shape[0]
    for key in strategy.keys():
        buys_idx, sells_idx = get_entry_and_exit_idx(buys_dict[key], sells_dict[key], N)
        buys_idx_dict[key] = buys_idx
        sells_idx_dict[key] = sells_idx

    wealths = np.ones((N,))

    if buys_idx_dict[key_spread][0] < sells_idx_dict[key_spread][0]:
        buy_state = True
        entry_i = buys_idx_dict[key_spread][0]
    else:
        buy_state = False
        entry_i = sells_idx_dict[key_spread][0]

    entry_is_transaction = True
    take_profit_triggered = False

    while entry_i < N - 1:
        exit_is_transaction = True
        if buy_state:
            larger_exits_idx = sells_idx_dict[key_spread][sells_idx_dict[key_spread] > entry_i]
        else:
            larger_exits_idx = buys_idx_dict[key_spread][buys_idx_dict[key_spread] > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        if exit_i - 1 - entry_i > aggregate_N * 60:
            exit_i = entry_i + aggregate_N * 60 + 1
            exit_is_transaction = False

        if buy_state:
            price = X[entry_i, 0]
            sub_X = X[entry_i:exit_i, :] / X[entry_i, 0]
        else:
            price = X_bear[entry_i, 0]
            sub_X = X_bear[entry_i:exit_i, :] / X_bear[entry_i, 0]

        old_wealth = wealths[entry_i]

        if entry_is_transaction:
            old_wealth *= commissions_and_spread

            buy_price = price
            buy_i = entry_i
            take_profit = take_profit_long if buy_state else take_profit_short
            take_profit_price = buy_price * take_profit
            take_profit_triggered = False

        li_take_profit = sub_X[1:, 1] * price > take_profit_price

        stop_loss_triggered = False
        stop_loss = stop_loss_long if buy_state else stop_loss_short
        li_stop_loss = sub_X[1:, 2] < stop_loss

        # check if take_profit or stop_loss is triggered
        if np.any(li_take_profit) and not take_profit_triggered:
            idx = np.arange(len(sub_X) - 1) + 1

            idx_take_profit = idx[li_take_profit]
            min_i_take_profit = idx_take_profit[0]

            take_profit_triggered = True
            spread = potential_spreads[np.argmin(np.abs(potential_balances - take_profit_price / price * old_wealth * total_balance))]
            commissions_and_spread = (1 - commissions - spread)

            wealths[entry_i:entry_i + min_i_take_profit] = sub_X[:min_i_take_profit, 0] * old_wealth
            wealths[entry_i + min_i_take_profit:exit_i] = take_profit_price / price * old_wealth * commissions_and_spread

        elif np.any(li_stop_loss):
            idx = np.arange(len(sub_X) - 1) + 1

            idx_stop_loss = idx[li_stop_loss]
            min_i_stop_loss = idx_stop_loss[0]

            stop_loss_triggered = True
            spread = potential_spreads[np.argmin(np.abs(potential_balances - stop_loss * old_wealth * total_balance))]
            commissions_and_spread = (1 - commissions - spread)

            wealths[entry_i:entry_i + min_i_stop_loss] = sub_X[:min_i_stop_loss, 0] * old_wealth
            wealths[entry_i + min_i_stop_loss:exit_i] = stop_loss * old_wealth * commissions_and_spread

        else:
            if take_profit_triggered:
                wealths[entry_i:exit_i] = old_wealth
            else:
                wealths[entry_i:exit_i] = sub_X[:, 0] * old_wealth

        spread = potential_spreads[np.argmin(np.abs(potential_balances - wealths[exit_i - 1] * total_balance))]
        commissions_and_spread = (1 - commissions - spread)

        prev_key_spread = key_spread
        key_spread = str(key_spreads_float[np.argmin(np.abs(key_spreads_float - spread))])

        if prev_key_spread != key_spread:
            aggregate_N, w, take_profit_long, take_profit_short, stop_loss_long, stop_loss_short = strategy[key_spread]['params']

            if buy_state:
                if buys_dict[prev_key_spread][exit_i - 1] != buys_dict[key_spread][exit_i - 1]:
                    exit_is_transaction = True
            else:
                if sells_dict[prev_key_spread][exit_i - 1] != sells_dict[key_spread][exit_i - 1]:
                    exit_is_transaction = True

            # update take_profit price
            if not exit_is_transaction:
                take_profit = take_profit_long if buy_state else take_profit_short
                take_profit_price = min(take_profit_price, buy_price * take_profit)

        if exit_is_transaction and not (take_profit_triggered or stop_loss_triggered):
            wealths[exit_i - 1] *= commissions_and_spread

        entry_i = exit_i - 1
        if exit_is_transaction:
            buy_state = not buy_state
            entry_is_transaction = True
        else:
            entry_is_transaction = False

        if stop_loss_triggered:
            entry_is_transaction = True

    return wealths




def apply_commissions_and_spreads_fast(wealth, N_transactions, commissions, spread, n_months = 1, from_log = False):
    commissions_and_spread = (1 - commissions - spread)

    if from_log:
        res = wealth + np.log(commissions_and_spread) * N_transactions
        return np.exp(res / n_months)

    res = wealth * commissions_and_spread ** N_transactions
    return np.exp(np.log(res) / n_months)


def apply_commissions_and_spreads(wealths, buys, sells, N, commissions, spread, inplace = True):
    buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)

    if not inplace:
        wealths = np.copy(wealths)

    if buys_idx[0] < sells_idx[0]:
        buy_state = True
        entry_i = buys_idx[0]
    else:
        buy_state = False
        entry_i = sells_idx[0]

    commissions_and_spread = (1 - commissions - spread)

    while entry_i < N - 1:
        if buy_state:
            larger_exits_idx = sells_idx[sells_idx > entry_i]
        else:
            larger_exits_idx = buys_idx[buys_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N

        wealths[entry_i:] *= commissions_and_spread
        wealths[exit_i - 1:] *= commissions_and_spread

        entry_i = exit_i - 1
        buy_state = not buy_state

    return wealths


# TODO: add min_trade_wealths
def get_trade_wealths(X, entries, exits, N):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    trade_wealths = []
    max_trade_wealths = []
    min_trade_wealths = []

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_X = X[entry_i:exit_i, :]
        if sub_X.shape[0] > 1:
            trade_wealths.append(sub_X[-1, 0] / sub_X[0, 0])
            max_trade_wealths.append(np.max(sub_X[1:, 1]) / sub_X[0, 0]) # use highs
            min_trade_wealths.append(np.min(sub_X[1:, 2]) / sub_X[0, 0]) # use lows

    return trade_wealths, min_trade_wealths, max_trade_wealths


def append_trade_wealths(wealths, entries, exits, N, trade_wealths, max_trade_wealths):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_wealths = wealths[entry_i:exit_i]
        trade_wealths.append(sub_wealths[-1] / sub_wealths[0])
        max_trade_wealths.append(sub_wealths.max() / sub_wealths[0])

def get_sub_trade_wealths(X, entries, exits, N):
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    sub_trade_wealths = []
    max_sub_trade_wealths = []
    min_sub_trade_wealths = []

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
        sub_X = X[entry_i:exit_i, :]
        sub_wealths = sub_X[1:, 0] / sub_X[0, 0]
        max_sub_wealths = sub_X[1:, 1] / sub_X[0, 0]
        min_sub_wealths = sub_X[1:, 2] / sub_X[0, 0]
        if sub_wealths.size > 0:
            sub_trade_wealths.append(sub_wealths)
            max_sub_trade_wealths.append(max_sub_wealths)
            min_sub_trade_wealths.append(min_sub_wealths)

    return sub_trade_wealths, min_sub_trade_wealths, max_sub_trade_wealths

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


def get_take_profit_wealths_from_trades(trade_wealths, max_trade_wealths, take_profit, total_months, commissions, spread, return_total = True, return_as_log = False):
    trade_wealths = np.array(trade_wealths)
    max_trade_wealths = np.array(max_trade_wealths)
    trade_wealths[max_trade_wealths > take_profit] = take_profit * (1 - commissions - spread)

    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return res if return_as_log else np.exp(res)

    return np.log(trade_wealths) if return_as_log else trade_wealths

def get_stop_loss_wealths_from_trades(trade_wealths, min_trade_wealths, stop_loss, total_months, commissions, spread, return_total = True, return_as_log = False):
    trade_wealths = np.array(trade_wealths)
    min_trade_wealths = np.array(min_trade_wealths)
    trade_wealths[min_trade_wealths < stop_loss] = stop_loss * (1 - commissions - spread)

    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return res if return_as_log else np.exp(res)

    return np.log(trade_wealths) if return_as_log else trade_wealths


def get_trailing_stop_loss_wealths_from_sub_trades(sub_trade_wealths, min_sub_trade_wealths, max_sub_trade_wealths, trailing_stop_loss, total_months, commissions, spread, return_total = True, return_as_log = False):
    trade_wealths = np.array([
        sub_trade_wealths[i][-1] for i in range(len(sub_trade_wealths))
    ])

    for i in range(len(sub_trade_wealths)):
        # price starts exactly at 1, otherwise would need to multiply with it
        trailing_diff = 1 - trailing_stop_loss

        # assume the worst case for trailing stop loss
        accumulative_max = np.concatenate([
            np.array([1.0]),
            np.maximum.accumulate(max_sub_trade_wealths[i])
        ])[:-1]
        li = min_sub_trade_wealths[i] < accumulative_max - trailing_diff

        if np.any(li):
            trade_wealths[i] = (accumulative_max[li][0] - trailing_diff) * (1 - commissions - spread)

    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return res if return_as_log else np.exp(res)

    return np.log(trade_wealths) if return_as_log else trade_wealths




def get_stop_loss_wealths_from_sub_trades(sub_trade_wealths, min_sub_trade_wealths, stop_loss, total_months, commissions, spread, return_total = True, return_as_log = False, return_N_transactions = False):
    trade_wealths = []
    N_transactions = 0
    for i in range(len(sub_trade_wealths)):
        sub_trade_wealths_i = np.copy(sub_trade_wealths[i])
        li = min_sub_trade_wealths[i] < stop_loss
        sub_trade_wealths_i[li] = stop_loss * (1 - commissions - spread) ** 2
        N_transactions += np.sum(li) * 2
        trade_wealths.append(np.prod(sub_trade_wealths_i) * (1 - commissions - spread) ** 2)

    trade_wealths = np.array(trade_wealths)
    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return_tuple = (res,) if return_as_log else (np.exp(res),)
    else:
        return_tuple = (np.log(trade_wealths),) if return_as_log else (trade_wealths,)

    if return_N_transactions:
        return_tuple = return_tuple + (N_transactions,)

    return return_tuple

def transform_wealths(wealths, X, entries, exits, N, take_profit, stop_loss, commissions, spread):
    assert(take_profit == np.Inf or stop_loss == 0)
    entries_idx, exits_idx = get_entry_and_exit_idx(entries, exits, N)

    for entry_i in entries_idx:
        larger_exits_idx = exits_idx[exits_idx > entry_i]
        exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N

        # Process take_profit
        if take_profit != np.Inf:
            # sub_wealths = wealths[entry_i:exit_i]
            li = X[entry_i+1:exit_i, 1] / X[entry_i, 0] > take_profit
            if np.any(li):
                first_i = np.arange(entry_i+1, exit_i)[li][0]
                old_wealth = wealths[exit_i - 1]
                wealths[first_i:exit_i] = wealths[entry_i+1] * take_profit * (1 - commissions - spread)
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
