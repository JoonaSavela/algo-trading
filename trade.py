from keys import ftx_api_key, ftx_secret_key
# from model import build_model
from ftx.rest.client import FtxClient
import time
from datetime import datetime
from data import coins, get_recent_data
import numpy as np
import pandas as pd
from utils import aggregate as aggregate_F
from utils import get_buys_and_sells, floor_to_n
from math import floor, log10
import json
from strategy import *
from parameters import params

target_symbol = 'ETH'
bull_symbol = target_symbol
bear_symbol = target_symbol + 'BEAR'
source_symbol = 'USDT'

# TODO: cleaner way? recall pandas usage, e.g. Data science handbook/cookbook
def ftx_price(client, symbol):
    markets = pd.DataFrame(client.list_markets())
    res = float(markets[(markets['baseCurrency'] == symbol) & (markets['quoteCurrency'] == source_symbol)]['ask'])
    time.sleep(0.05)
    return res

def asset_balance(client, symbol, in_usd = False):
    balances = pd.DataFrame(client.get_balances())
    time.sleep(0.05)
    balances = balances.set_index('coin')['usdValue' if in_usd else 'free']
    if isinstance(symbol, list):
        res = tuple(map(lambda x: balances.get(x, 0), symbol))
        if len(res) == 1:
            res = res[0]
    else:
        res = balances.get(symbol, 0)

    return res


def get_total_balance(client, separate = True):
    balances = pd.DataFrame(client.get_balances()).set_index('coin')
    time.sleep(0.05)
    total_balance = balances['usdValue'].sum()

    if separate:
        return total_balance, balances

    return total_balance

def sell_assets(client, symbol, amount = 1, round_n = 4):
    balance = asset_balance(client, symbol)
    symbol += '/' +  source_symbol
    quantity = floor_to_n(balance * amount, round_n)
    client.place_order(
        market = symbol,
        side = 'sell',
        price = None,
        size = quantity,
        type = 'market'
    )
    time.sleep(0.1)

def buy_assets(client, symbol, amount = 1, round_n = 4):
    prev_high = ftx_price(client, symbol)
    symbol += '/' +  source_symbol
    balance_source = asset_balance(client, source_symbol, True)

    quantity = floor_to_n(balance_source / prev_high * amount, round_n)

    client.place_order(
        market = symbol,
        side = 'buy',
        price = None,
        size = quantity,
        type = 'market'
    )
    time.sleep(0.1)



def buy_assets_short(client, m, relative_balance_short, min_amount):
    if relative_balance_short > min_amount:
        sell_assets(
            client,
            bear_symbol,
            round_n = 4
        )
    time.sleep(1)

    buy_assets(
        client,
        bull_symbol,
        amount = m if m <= 1 else m / 3,
        round_n = 5 if m <= 1 else 4
    )
    time.sleep(1)

def sell_assets_short(client, m, m_bear, relative_balance_target, min_amount):
    if relative_balance_target > min_amount:
        sell_assets(
            client,
            bull_symbol,
            round_n = 5 if m <= 1 else 4
        )
    time.sleep(1)

    buy_assets(
        client,
        bear_symbol,
        amount = m_bear / 3,
        round_n = 4
    )
    time.sleep(1)

def place_take_profit(client, symbol, take_profit):
    price = ftx_price(client, symbol)
    balance = float(asset_balance(client, symbol))
    symbol += '/' +  source_symbol

    client.place_conditional_order(
        market = symbol,
        side = 'sell',
        size = balance,
        order_type = 'takeProfit',
        trigger_price = price * take_profit
    )
    time.sleep(0.05)


def wait(time_delta, initial_time, verbose = False):
    time_diff = time.time() - initial_time
    waiting_time = time_delta - time_diff
    while waiting_time < 0:
        waiting_time += time_delta
    if verbose:
        print()
        if waiting_time / (60 * 60) < 1:
            print('waiting', waiting_time / 60, 'minutes')
        else:
            print('waiting', waiting_time / (60 * 60), 'hours')
    time.sleep(waiting_time)

def cancel_orders(client):
    client.cancel_orders()
    time.sleep(0.05)


# TODO: move all changeable parameters into a text/json file(s)
def trading_pipeline(buy_flag, sell_flag):
    print('Starting trading pipeline...')

    global bull_symbol
    client = FtxClient(ftx_api_key, ftx_secret_key)
    min_amount = 0.01
    print(target_symbol)

    params_dict = params[target_symbol]
    print(params_dict)
    aggregate = params_dict['aggregate']
    w = params_dict['w']
    m = params_dict['m']
    m_bear = params_dict['m_bear']
    take_profit_long = params_dict['take_profit_long']
    take_profit_short = params_dict['take_profit_short']
    displacement = params_dict['displacement']

    debug_flag = False

    if buy_flag is None and sell_flag is None:
        buy_flag = False
        sell_flag = False

        flag = input('Which event has happened? ')
        if flag == 'buy':
            buy_flag = True
        elif flag == 'sell':
            sell_flag = True

        debug = input('Is this a debug run? ')
        if 'y' in debug:
            debug_flag = True

        if debug_flag:
            print('Debug flag set')

    if m > 1:
        bull_symbol += 'BULL'

    time_delta = 60 * 60 * aggregate

    error_flag = True

    try:
        total_balance, balances = get_total_balance(client, True)
        print()
        print(balances)

        while True:
            # rounds the current time into the previous hour,
            # or the next hour if the displacement minute has passed
            seconds = time.time() + (59 - displacement) * 60
            localtime = time.localtime(seconds)
            now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
            timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
            wait(displacement * 60, timeTo, verbose = debug_flag)

            X, _ = get_recent_data(target_symbol, size = (w + 1) * 60 * aggregate, type='m')
            X = aggregate_F(X, aggregate, type = 'h')

            balance_target, balance_short, balance_source = asset_balance(
                client,
                [bull_symbol, bear_symbol, source_symbol],
                True
            )
            total_balance = sum([balance_target, balance_short, balance_source])

            buy, sell, _ = get_buys_and_sells(X, w, True)
            if debug_flag:
                print(buy, sell)

            price = X[-1, 0]
            action = 'DO NOTHING'

            if buy and not buy_flag and not debug_flag and (
                    balance_source / total_balance > min_amount or
                    balance_short / total_balance > min_amount
                ):
                cancel_orders(client)
                buy_assets_short(client, m, balance_short / total_balance, min_amount)
                place_take_profit(client, bull_symbol, take_profit_long)
                action = 'BUY'
                buy_flag = True
                sell_flag = False
            elif sell and not sell_flag and not debug_flag and (
                    balance_target / total_balance > min_amount or
                    balance_source / total_balance > min_amount
                ):
                cancel_orders(client)
                sell_assets_short(client, m, m_bear, balance_target / total_balance, min_amount)
                place_take_profit(client, bear_symbol, take_profit_short)
                action = 'SELL'
                buy_flag = False
                sell_flag = True


            if action != 'DO NOTHING':
                print()
                print(timeTo, action, price)
                total_balance, balances = get_total_balance(client, True)
                print(balances)
                print()

            wait(time_delta, timeTo, verbose = debug_flag)


    except KeyboardInterrupt:
        error_flag = False
    finally:
        print()
        print('Exiting trading pipeline...')

        if error_flag:
            cancel_orders(client)
            if buy_flag:
                sell_assets(
                    client,
                    bull_symbol,
                    amount = 0.5,
                    round_n = 5 if m <= 1 else 4
                )
                time.sleep(1)
                place_take_profit(client, bull_symbol, take_profit_long)

            elif sell_flag:
                sell_assets(
                    client,
                    bear_symbol,
                    amount = 0.5,
                    round_n = 4
                )
                time.sleep(1)
                place_take_profit(client, bear_symbol, take_profit_short)

        # cancel_orders(client)
        total_balance, balances = get_total_balance(client, True)
        print()
        print(balances)

        return error_flag, buy_flag, sell_flag

if __name__ == '__main__':
    error_flag = True
    buy_flag = None
    sell_flag = None

    while error_flag:
        if buy_flag is not None and sell_flag is not None:
            time.sleep(60)
        error_flag, buy_flag, sell_flag = trading_pipeline(buy_flag, sell_flag)
