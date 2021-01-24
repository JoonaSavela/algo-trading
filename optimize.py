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


def return_trade_wealths(trade_wealths, total_months, taxes = True, return_total = True, return_as_log = True):
    if taxes:
        trade_wealths = apply_taxes(trade_wealths)

    if return_total:
        res = np.sum(np.log(trade_wealths)) / total_months
        return res if return_as_log else np.exp(res)

    return np.log(trade_wealths) if return_as_log else trade_wealths




# TODO: add comments or split this into multiple functions
def get_adaptive_wealths(
    X,
    buys,
    sells,
    aggregate_N,
    sltp_args,
    total_balance,
    potential_balances,
    potential_spreads,
    place_take_profit_and_stop_loss_simultaneously = False,
    trail_value_recalc_period = None,
    commissions = 0.0007,
    short = False,
    X_bear = None,
    taxes = True,
    ):

    # TODO: move this into a separate function?
    spread = potential_spreads[np.argmin(np.abs(potential_balances - total_balance))]
    commissions_and_spread = (1 - commissions - spread)

    # assert(short == (len(sltp_args) == 4))
    assert(short == (X_bear is not None))

    N = buys.shape[0]
    buys_idx, sells_idx = get_entry_and_exit_idx(buys, sells, N)

    if trail_value_recalc_period is None:
        trail_value_recalc_period = N + 1

    wealths = np.ones((N,))

    if buys_idx[0] < sells_idx[0]:
        buy_state = True
        entry_i = buys_idx[0]
    else:
        buy_state = False
        entry_i = sells_idx[0]

    entry_is_transaction = buy_state or short

    # handle place_take_profit_and_stop_loss_simultaneously = True
    if place_take_profit_and_stop_loss_simultaneously:
        trigger_names_and_params = {}
        trigger_names_and_params[True] = sltp_args[1:4]
        if short:
            trigger_names_and_params[False] = sltp_args[5:]

        trigger_triggered = False

        while entry_i < N - 1:
            exit_is_transaction = True
            if buy_state:
                larger_exits_idx = sells_idx[sells_idx > entry_i]
            else:
                larger_exits_idx = buys_idx[buys_idx > entry_i]
            exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
            if (exit_i - 1 - entry_i > aggregate_N * 60) and (buy_state or short):
                exit_i = entry_i + aggregate_N * 60 + 1
                exit_is_transaction = False

            if buy_state:
                price = X[entry_i, 0]
                sub_X = X[entry_i:exit_i, :] / X[entry_i, 0]
            elif short:
                price = X_bear[entry_i, 0]
                sub_X = X_bear[entry_i:exit_i, :] / X_bear[entry_i, 0]

            old_wealth = wealths[entry_i]

            if not (buy_state or short):
                wealths[entry_i:exit_i] = old_wealth
            else:
                if entry_is_transaction:
                    old_wealth *= commissions_and_spread

                    buy_price = price
                    buy_i = entry_i

                    take_profit, stop_loss_type, stop_loss = trigger_names_and_params[buy_state]
                    take_profit_price = buy_price * take_profit
                    stop_loss_price = buy_price * stop_loss

                    trigger_triggered = False
                    max_price = price

                    trail_value_recalc_idx = np.arange(buy_i + 1, N, trail_value_recalc_period)

                if not trigger_triggered:
                    take_profit_prices = np.repeat(take_profit_price, exit_i - entry_i - 1)

                    if stop_loss_type != 'trailing':
                        stop_loss_prices1 = stop_loss_prices2 = np.repeat(stop_loss_price, exit_i - entry_i - 1)
                    else:
                        li = (trail_value_recalc_idx > entry_i + 1) & (trail_value_recalc_idx < exit_i)
                        starts_and_ends = trail_value_recalc_idx[li]
                        starts_and_ends = np.concatenate([np.array([entry_i + 1]), starts_and_ends, np.array([exit_i])])

                        stop_loss_prices1 = np.zeros((exit_i - entry_i - 1,))
                        stop_loss_prices2 = np.zeros((exit_i - entry_i - 1,))

                        for start, end in zip(starts_and_ends[:-1], starts_and_ends[1:]):
                            if (start - buy_i - 1) % trail_value_recalc_period == 0:
                                acc_max_start_price = sub_X[start - entry_i - 1, 0] * price

                                if start - buy_i - 1 == 0:
                                    trail_value = (stop_loss - 1.0) * buy_price
                                else:
                                    trail_value = stop_loss_prices2[start - entry_i - 2] - sub_X[start - entry_i - 1, 0] * price
                            else:
                                acc_max_start_price = max_price

                            accumulative_max = np.maximum.accumulate(np.concatenate([
                                np.array([acc_max_start_price]),
                                sub_X[(start - entry_i):(end - entry_i), 1] * price
                            ]))
                            max_price = accumulative_max[-1]
                            # for each candle/timestep, low occurs before high
                            stop_loss_prices1[(start - entry_i - 1):(end - entry_i - 1)] = accumulative_max[:-1] + trail_value
                            # for each candle/timestep, high occurs before low
                            stop_loss_prices2[(start - entry_i - 1):(end - entry_i - 1)] = accumulative_max[1:] + trail_value

                        assert(np.all(np.diff(stop_loss_prices1) >= 0))
                        assert(np.all(np.diff(stop_loss_prices2) >= 0))

                    li_take_profit = sub_X[1:, 1] * price > take_profit_prices
                    cond_take_profit = np.any(li_take_profit)

                    li_stop_loss1 = sub_X[1:, 2] * price < stop_loss_prices1
                    if stop_loss_type == 'trailing':
                        li_stop_loss2 = sub_X[1:, 2] * price < stop_loss_prices2
                    else:
                        li_stop_loss2 = li_stop_loss1

                    cond_stop_loss1 = np.any(li_stop_loss1)
                    cond_stop_loss2 = np.any(li_stop_loss2)

                # check if trigger is triggered
                if not trigger_triggered and (cond_stop_loss1 or cond_stop_loss2 or cond_take_profit):
                    idx = np.arange(len(sub_X) - 1) + 1

                    if cond_take_profit:
                        i_take_profit = idx[li_take_profit][0]

                    if cond_stop_loss1:
                        idx1 = idx[li_stop_loss1]
                        i_stop_loss1 = idx1[0]
                        stop_loss_price1 = stop_loss_prices1[li_stop_loss1][0]
                        trigger_triggered = True
                    else:
                        i_stop_loss1 = len(sub_X) - 1
                        stop_loss_price1 = sub_X[-1, 0] * price

                    if stop_loss_type != 'trailing':
                        i_stop_loss = i_stop_loss1
                        stop_loss_price = stop_loss_price1

                    if stop_loss_type == 'trailing' and cond_stop_loss2:
                        idx2 = idx[li_stop_loss2]
                        i_stop_loss2 = idx2[0]

                        stop_loss_price2 = (max(stop_loss_prices1[li_stop_loss2][0], sub_X[i_stop_loss2, 2] * price) + stop_loss_prices2[li_stop_loss2][0]) / 2

                        possible_trigger_prices = np.array([stop_loss_price1, stop_loss_price2])
                        possible_trigger_prices_min_i = np.argmin(possible_trigger_prices)
                        stop_loss_price = possible_trigger_prices[possible_trigger_prices_min_i]

                        i_stop_losses = np.array([i_stop_loss1, i_stop_loss2])
                        i_stop_loss = i_stop_losses[possible_trigger_prices_min_i]

                        if i_stop_loss == i_stop_loss2:
                            trigger_triggered = True

                    if cond_stop_loss2:
                        if cond_take_profit:
                            if stop_loss_type == 'trailing':
                                # handle both cases of triggered trailing prices here (not just the minimum)
                                # because the max triggered trailing price is still smaller than take profit
                                if i_stop_loss <= i_take_profit:
                                    trigger_price = stop_loss_price
                                    min_i = i_stop_loss
                                elif i_stop_losses[1 - possible_trigger_prices_min_i] <= i_take_profit:
                                    trigger_price = possible_trigger_prices[1 - possible_trigger_prices_min_i]
                                    min_i = i_stop_losses[1 - possible_trigger_prices_min_i]
                                else:
                                    trigger_price = take_profit_price
                                    trigger_triggered = True
                                    min_i = i_take_profit
                            else:
                                # cond_stop_loss1 is equal to cond_stop_loss2
                                if i_stop_loss <= i_take_profit:
                                    trigger_price = stop_loss_price
                                    min_i = i_stop_loss
                                else:
                                    trigger_price = take_profit_price
                                    trigger_triggered = True
                                    min_i = i_take_profit
                        else:
                            trigger_price = stop_loss_price
                            min_i = i_stop_loss
                    elif cond_take_profit:
                        trigger_price = take_profit_price
                        trigger_triggered = True
                        min_i = i_take_profit

                    wealths[entry_i:entry_i + min_i] = sub_X[:min_i, 0] * old_wealth

                    new_wealth = trigger_price / price * old_wealth
                    if trigger_price > buy_price and taxes:
                        profit = trigger_price / buy_price
                        new_wealth *= apply_taxes(profit, copy = True) / profit

                    wealths[entry_i + min_i:exit_i] = new_wealth

                    if trigger_triggered:
                        spread = potential_spreads[np.argmin(np.abs(potential_balances - new_wealth * total_balance))]
                        commissions_and_spread = (1 - commissions - spread)

                        wealths[entry_i + min_i:exit_i] *= commissions_and_spread

                elif trigger_triggered:
                    wealths[entry_i:exit_i] = old_wealth
                else:
                    wealths[entry_i:exit_i] = sub_X[:, 0] * old_wealth

            if exit_is_transaction and not trigger_triggered and (buy_state or short) and taxes:
                sell_price = sub_X[-1, 0] * price
                if sell_price > buy_price:
                    profit = sell_price / buy_price
                    wealths[exit_i - 1] *= apply_taxes(profit, copy = True) / profit

            spread = potential_spreads[np.argmin(np.abs(potential_balances - wealths[exit_i - 1] * total_balance))]
            commissions_and_spread = (1 - commissions - spread)

            if exit_is_transaction and not trigger_triggered and (buy_state or short):
                wealths[exit_i - 1] *= commissions_and_spread

            entry_i = exit_i - 1
            if exit_is_transaction:
                buy_state = not buy_state
                entry_is_transaction = True
            else:
                entry_is_transaction = False

    else:
        trigger_names_and_params = {}
        trigger_names_and_params[True] = sltp_args[:2]
        if short:
            trigger_names_and_params[False] = sltp_args[2:]

        trigger_triggered = False

        while entry_i < N - 1:
            exit_is_transaction = True
            if buy_state:
                larger_exits_idx = sells_idx[sells_idx > entry_i]
            else:
                larger_exits_idx = buys_idx[buys_idx > entry_i]
            exit_i = larger_exits_idx[0] + 1 if larger_exits_idx.size > 0 else N
            if (exit_i - 1 - entry_i > aggregate_N * 60) and (buy_state or short):
                exit_i = entry_i + aggregate_N * 60 + 1
                exit_is_transaction = False

            if buy_state:
                price = X[entry_i, 0]
                sub_X = X[entry_i:exit_i, :] / X[entry_i, 0]
            elif short:
                price = X_bear[entry_i, 0]
                sub_X = X_bear[entry_i:exit_i, :] / X_bear[entry_i, 0]

            old_wealth = wealths[entry_i]

            if not (buy_state or short):
                wealths[entry_i:exit_i] = old_wealth
            else:
                if entry_is_transaction:
                    old_wealth *= commissions_and_spread

                    buy_price = price
                    buy_i = entry_i

                    trigger_name, trigger_param = trigger_names_and_params[buy_state]
                    trigger_price = buy_price * trigger_param
                    trigger_triggered = False
                    max_price = price

                    trail_value_recalc_idx = np.arange(buy_i + 1, N, trail_value_recalc_period)

                if not trigger_triggered:
                    if trigger_name != 'trailing':
                        trigger_prices1 = trigger_prices2 = np.repeat(trigger_price, exit_i - entry_i - 1)
                    else:
                        # TODO: adapt to the real world scenario where trail_value is updated
                        # (s.t. trigger price won't decrease), and accumulative_max is reset

                        li = (trail_value_recalc_idx > entry_i + 1) & (trail_value_recalc_idx < exit_i)
                        starts_and_ends = trail_value_recalc_idx[li]
                        starts_and_ends = np.concatenate([np.array([entry_i + 1]), starts_and_ends, np.array([exit_i])])

                        trigger_prices1 = np.zeros((exit_i - entry_i - 1,))
                        trigger_prices2 = np.zeros((exit_i - entry_i - 1,))

                        for start, end in zip(starts_and_ends[:-1], starts_and_ends[1:]):
                            if (start - buy_i - 1) % trail_value_recalc_period == 0:
                                acc_max_start_price = sub_X[start - entry_i - 1, 0] * price

                                if start - buy_i - 1 == 0:
                                    trail_value = (trigger_param - 1.0) * buy_price
                                else:
                                    trail_value = trigger_prices2[start - entry_i - 2] - sub_X[start - entry_i - 1, 0] * price
                            else:
                                acc_max_start_price = max_price

                            accumulative_max = np.maximum.accumulate(np.concatenate([
                                np.array([acc_max_start_price]),
                                sub_X[(start - entry_i):(end - entry_i), 1] * price
                            ]))
                            max_price = accumulative_max[-1]
                            # for each candle/timestep, low occurs before high
                            trigger_prices1[(start - entry_i - 1):(end - entry_i - 1)] = accumulative_max[:-1] + trail_value
                            # for each candle/timestep, high occurs before low
                            trigger_prices2[(start - entry_i - 1):(end - entry_i - 1)] = accumulative_max[1:] + trail_value

                        assert(np.all(np.diff(trigger_prices1) >= 0))
                        assert(np.all(np.diff(trigger_prices2) >= 0))

                    if trigger_name == 'take_profit':
                        li1 = li2 = sub_X[1:, 1] * price > trigger_prices1
                    else:
                        li1 = sub_X[1:, 2] * price < trigger_prices1
                        if trigger_name == 'trailing':
                            li2 = sub_X[1:, 2] * price < trigger_prices2
                        else:
                            li2 = li1

                    cond1 = np.any(li1)
                    cond2 = np.any(li2)

                # check if trigger is triggered
                if not trigger_triggered and (cond1 or cond2):
                    idx = np.arange(len(sub_X) - 1) + 1

                    if cond1:
                        idx1 = idx[li1]
                        min_i = idx1[0]
                        trigger_price = trigger_prices1[li1][0]
                        trigger_triggered = True
                    else:
                        min_i = len(sub_X) - 1
                        trigger_price = sub_X[-1, 0] * price

                    if trigger_name == 'trailing':
                        idx2 = idx[li2]
                        min_i2 = idx2[0]

                        prev_trigger_price = trigger_price
                        trigger_price = np.min([
                            trigger_price,
                            (max(trigger_prices1[li2][0], sub_X[min_i2, 2] * price) + trigger_prices2[li2][0]) / 2
                        ])

                        if trigger_price != prev_trigger_price:
                            trigger_triggered = True
                            min_i = min_i2

                    wealths[entry_i:entry_i + min_i] = sub_X[:min_i, 0] * old_wealth

                    new_wealth = trigger_price / price * old_wealth
                    if trigger_price > buy_price and taxes:
                        profit = trigger_price / buy_price
                        new_wealth *= apply_taxes(profit, copy = True) / profit

                    wealths[entry_i + min_i:exit_i] = new_wealth

                    if trigger_triggered:
                        spread = potential_spreads[np.argmin(np.abs(potential_balances - new_wealth * total_balance))]
                        commissions_and_spread = (1 - commissions - spread)

                        wealths[entry_i + min_i:exit_i] *= commissions_and_spread

                elif trigger_triggered:
                    wealths[entry_i:exit_i] = old_wealth
                else:
                    wealths[entry_i:exit_i] = sub_X[:, 0] * old_wealth

            if exit_is_transaction and not trigger_triggered and (buy_state or short) and taxes:
                sell_price = sub_X[-1, 0] * price
                if sell_price > buy_price:
                    profit = sell_price / buy_price
                    wealths[exit_i - 1] *= apply_taxes(profit, copy = True) / profit

            spread = potential_spreads[np.argmin(np.abs(potential_balances - wealths[exit_i - 1] * total_balance))]
            commissions_and_spread = (1 - commissions - spread)

            if exit_is_transaction and not trigger_triggered and (buy_state or short):
                wealths[exit_i - 1] *= commissions_and_spread

            entry_i = exit_i - 1
            if exit_is_transaction:
                buy_state = not buy_state
                entry_is_transaction = True
            else:
                entry_is_transaction = False


    return wealths




def apply_commissions_and_spreads(wealth, N_transactions, commissions, spread, n_months = 1, from_log = False):
    commissions_and_spread = (1 - commissions - spread)

    if from_log:
        res = wealth + np.log(commissions_and_spread) * N_transactions
        return np.exp(res / n_months)

    res = wealth * commissions_and_spread ** N_transactions
    return np.exp(np.log(res) / n_months)


# TODO: rename "trade_wealths"
# No commissions or spread
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


# No commissions or spread
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




# No commissions or spread
def get_take_profit_wealths_from_trades(trade_wealths, max_trade_wealths, take_profit, total_months, return_total = True, return_as_log = False, taxes = True):
    trade_wealths = np.array(trade_wealths)
    max_trade_wealths = np.array(max_trade_wealths)
    trade_wealths[max_trade_wealths > take_profit] = take_profit

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes = taxes,
        return_total = return_total,
        return_as_log = return_as_log
    )

# No commissions or spread
def get_stop_loss_wealths_from_trades(trade_wealths, min_trade_wealths, stop_loss, total_months, return_total = True, return_as_log = False, taxes = True):
    trade_wealths = np.array(trade_wealths)
    min_trade_wealths = np.array(min_trade_wealths)
    trade_wealths[min_trade_wealths < stop_loss] = stop_loss

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes = taxes,
        return_total = return_total,
        return_as_log = return_as_log
    )

# No commissions or spread
def get_trailing_stop_loss_wealths_from_sub_trades(
    sub_trade_wealths,
    min_sub_trade_wealths,
    max_sub_trade_wealths,
    trailing_stop_loss,
    total_months,
    trail_value_recalc_period = None,
    return_total = True,
    return_as_log = False,
    taxes = True
    ):
    trade_wealths = np.array([
        sub_trade_wealths[i][-1] for i in range(len(sub_trade_wealths))
    ])

    if trail_value_recalc_period is None:
        trail_value_recalc_period = np.max([len(x) for x in sub_trade_wealths]) + 1

    for i in range(len(sub_trade_wealths)):
        n = len(sub_trade_wealths[i])
        starts = np.arange(0, n, trail_value_recalc_period)
        ends =  np.concatenate([starts[1:], np.array([n])])
        starts_and_ends = np.stack([starts, ends], axis = -1)

        trigger_prices1 = np.zeros((n,))
        trigger_prices2 = np.zeros((n,))

        for start, end in starts_and_ends:
            # update trail_value like in the real world (trigger price won't decrease)
            if start == 0:
                trail_value = trailing_stop_loss - 1.0
            else:
                trail_value = trigger_prices2[start - 1] - sub_trade_wealths[i][start - 1]

            # accumulative_max is reset
            accumulative_max = np.maximum.accumulate(np.concatenate([
                np.array([1.0 if start == 0 else sub_trade_wealths[i][start - 1]]),
                max_sub_trade_wealths[i][start:end]
            ]))

            # for each candle/timestep, low occurs before high
            trigger_prices1[start:end] = accumulative_max[:-1] + trail_value
            # for each candle/timestep, high occurs before low
            trigger_prices2[start:end] = accumulative_max[1:] + trail_value

        assert(np.all(np.diff(trigger_prices1) >= 0))
        assert(np.all(np.diff(trigger_prices2) >= 0))

        li1 = min_sub_trade_wealths[i] < trigger_prices1
        li2 = min_sub_trade_wealths[i] < trigger_prices2

        cond1 = np.any(li1)
        cond2 = np.any(li2)

        # not possible to have cond1 == True and cond2 == False
        assert(not (cond1 == True and cond2 == False))

        if cond2:
            # assume the worst case
            trade_wealths[i] = np.min([
                trigger_prices1[li1][0] if cond1 else trade_wealths[i],
                (max(trigger_prices1[li2][0], min_sub_trade_wealths[i][li2][0]) + trigger_prices2[li2][0]) / 2
            ])

    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes = taxes,
        return_total = return_total,
        return_as_log = return_as_log
    )


# No commissions or spread
def get_stop_loss_take_profit_wealths_from_sub_trades(
    sub_trade_wealths,
    min_sub_trade_wealths,
    max_sub_trade_wealths,
    stop_loss,
    stop_loss_type_is_trailing,
    take_profit,
    total_months,
    trail_value_recalc_period = None,
    return_total = True,
    return_as_log = False,
    taxes = True
    ):
    trade_wealths = np.array([
        sub_trade_wealths[i][-1] for i in range(len(sub_trade_wealths))
    ])

    if trail_value_recalc_period is None:
        trail_value_recalc_period = np.max([len(x) for x in sub_trade_wealths]) + 1

    for i in range(len(sub_trade_wealths)):
        n = len(sub_trade_wealths[i])
        idx = np.arange(n)

        li_take_profit = max_sub_trade_wealths[i] > take_profit
        cond_take_profit = np.any(li_take_profit)

        if cond_take_profit:
            i_take_profit = idx[li_take_profit][0]


        if stop_loss_type_is_trailing:
            starts = np.arange(0, n, trail_value_recalc_period)
            ends =  np.concatenate([starts[1:], np.array([n])])
            starts_and_ends = np.stack([starts, ends], axis = -1)

            trigger_prices1 = np.zeros((n,))
            trigger_prices2 = np.zeros((n,))

            for start, end in starts_and_ends:
                # update trail_value like in the real world (trigger price won't decrease)
                if start == 0:
                    trail_value = stop_loss - 1.0
                else:
                    trail_value = trigger_prices2[start - 1] - sub_trade_wealths[i][start - 1]

                # accumulative_max is reset
                accumulative_max = np.maximum.accumulate(np.concatenate([
                    np.array([1.0 if start == 0 else sub_trade_wealths[i][start - 1]]),
                    max_sub_trade_wealths[i][start:end]
                ]))

                # for each candle/timestep, low occurs before high
                trigger_prices1[start:end] = accumulative_max[:-1] + trail_value
                # for each candle/timestep, high occurs before low
                trigger_prices2[start:end] = accumulative_max[1:] + trail_value

            assert(np.all(np.diff(trigger_prices1) >= 0))
            assert(np.all(np.diff(trigger_prices2) >= 0))

            li_stop_loss1 = min_sub_trade_wealths[i] < trigger_prices1
            li_stop_loss2 = min_sub_trade_wealths[i] < trigger_prices2

            cond_stop_loss1 = np.any(li_stop_loss1)
            cond_stop_loss2 = np.any(li_stop_loss2)

            # not possible to have cond_stop_loss1 == True and cond_stop_loss2 == False
            assert(not (cond_stop_loss1 == True and cond_stop_loss2 == False))

            if cond_stop_loss2:
                i_stop_loss1 = idx[li_stop_loss1][0] if cond_stop_loss1 else n + 1
                i_stop_loss2 = idx[li_stop_loss2][0]
                idx_stop_loss = np.array([i_stop_loss1, i_stop_loss2])

                # assume the worst case
                possible_trigger_prices = np.array([
                    trigger_prices1[i_stop_loss1] if cond_stop_loss1 else trade_wealths[i],
                    (max(trigger_prices1[i_stop_loss2], min_sub_trade_wealths[i][i_stop_loss2]) + trigger_prices2[i_stop_loss2]) / 2
                ])

                possible_trigger_prices_min_i = np.argmin(possible_trigger_prices)

                stop_loss_trigger_price = possible_trigger_prices[possible_trigger_prices_min_i]
                i_stop_loss = idx_stop_loss[possible_trigger_prices_min_i]
        else:
            li_stop_loss = min_sub_trade_wealths[i] < stop_loss
            cond_stop_loss2 = np.any(li_stop_loss)
            stop_loss_trigger_price = stop_loss
            if cond_stop_loss2:
                i_stop_loss = idx[li_stop_loss][0]

        if cond_stop_loss2:
            if cond_take_profit:
                if stop_loss_type_is_trailing:
                    # handle both cases of triggered trailing prices here (not just the minimum)
                    # because the max triggered trailing price is still smaller than take profit
                    if i_stop_loss <= i_take_profit:
                        trade_wealths[i] = stop_loss_trigger_price
                    elif idx_stop_loss[1 - possible_trigger_prices_min_i] <= i_take_profit:
                        trade_wealths[i] = possible_trigger_prices[1 - possible_trigger_prices_min_i]
                    else:
                        trade_wealths[i] = take_profit
                else:
                    if i_stop_loss <= i_take_profit:
                        trade_wealths[i] = stop_loss_trigger_price
                    else:
                        trade_wealths[i] = take_profit
            else:
                trade_wealths[i] = stop_loss_trigger_price
        elif cond_take_profit:
            trade_wealths[i] = take_profit


    return return_trade_wealths(
        trade_wealths,
        total_months,
        taxes = taxes,
        return_total = return_total,
        return_as_log = return_as_log
    )

# TODO: make a single function for calculating wealths
#   - both as separate trades and a full time series
#       - option to combine separate trades into one
#   - needs to handle spreads adaptively
#   - how to handle generating inputs for portfolio optimization (each strategy
#     performance needs to be the same length with exactly the same time stamps)?
#   - how to handle short side?

def combine_sub_trade_wealths(sub_trade_wealths_long, sub_trade_wealths_short = None):
    pass






if __name__ == '__main__':
    pass
