from keys import binance_api_key, binance_secret_key
# from model import build_model
import binance
from binance.client import Client
from binance.enums import *
import time
from data import coins, get_recent_data
import numpy as np
from utils import *
from math import floor, log10
import json
from strategy import *
from requests.exceptions import ConnectionError, ReadTimeout
from visualize import get_trades
from parameters import params

# TODO: detect which coins are owned

target_symbol = 'ETH'
source_symbol = 'USDT'

# TODO: start using OCO orders

# TODO: change to prev close
def binance_price(client, symbol):
    res = float(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)[-1][2])
    time.sleep(0.05)
    return res

def asset_balance(client, symbol, in_usd = False):
    res = float(client.get_asset_balance(asset=symbol)['free'])
    time.sleep(0.05)
    if in_usd and symbol != 'USDT':
        res *= binance_price(client, symbol + 'USDT')
    return res

def get_total_balance(client, separate = True):
    balance_target = asset_balance(client, target_symbol, True)
    balance_source = asset_balance(client, source_symbol, True)
    balance_bnb = asset_balance(client, 'BNB', True)
    total_balance = balance_target + balance_source + balance_bnb

    if separate:
        return total_balance, balance_target, balance_source, balance_bnb

    return total_balance

def check_bnb(client):
    total_balance, balance_target, balance_source, balance_bnb = get_total_balance(client)
    if balance_bnb / (total_balance) < 0.005:
        if balance_target > balance_source:
            symbol = target_symbol
        else:
            symbol = source_symbol
        try:
            symbol = 'BNB' + symbol
            client.order_market_buy(symbol=symbol, quantity=0.01 * total_balance)
            print('Bought BNB')
            time.sleep(0.1)
        except binance.exceptions.BinanceAPIException as e:
            print(e)

def snippet(amount, precision):
    return "{:0.0{}f}".format(amount, precision)

def cancel_orders(client, symbol, side = None):
    open_orders = client.get_open_orders(symbol=symbol)
    count = 0
    for order in open_orders:
        if order['side'] is None or order['side'] == side:
            result = client.cancel_order(symbol=symbol, orderId=order['orderId'])
            time.sleep(0.1)
            count += 1
    if count > 0:
        print('Cancelled', count, 'orders')

def sell_assets(client, amount = 1):
    symbol = target_symbol + source_symbol
    balance_target = asset_balance(client, target_symbol)
    quantity = floor_to_n(balance_target * amount, 4)
    try:
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity)
        time.sleep(0.1)
        return True
    except binance.exceptions.BinanceAPIException as e:
        print(e)
        return False

# TODO: use amount = 0.99?
def buy_assets(client, amount = 1):
    symbol = target_symbol + source_symbol
    prev_high = binance_price(client, symbol)
    balance_source = asset_balance(client, source_symbol, True)

    quantity = floor_to_n(balance_source / prev_high * amount, 4)
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity)
        time.sleep(0.1)
        return True
    except binance.exceptions.BinanceAPIException as e:
        print(e)
        return False

def wait(time_delta, initial_time, verbose = False):
    time_diff = time.time() - initial_time
    waiting_time = time_delta - time_diff
    while waiting_time < 0:
        waiting_time += time_delta
    if verbose:
        print()
        print('waiting', waiting_time / (60 * 60), 'hours')
    time.sleep(waiting_time)

def get_status(client):
    status = client.get_account_status()
    time.sleep(0.05)
    return status

def trading_pipeline():
    client = Client(binance_api_key, binance_secret_key)
    symbol = target_symbol + source_symbol
    print(target_symbol)

    params_dict = params[target_symbol]
    print(params_dict)
    type = params_dict['type']
    aggregate = params_dict['aggregate']
    w = params_dict['w']

    time_delta = 60 * 60 * aggregate

    try:
        print('Starting trading pipeline...')

        status = get_status(client)
        print('Status:', status)
        check_bnb(client)

        _, initial_time = get_recent_data(target_symbol, type='h', aggregate=aggregate)
        balance_capital, balance_target, balance_source, balance_bnb = get_total_balance(client)
        print()
        print(initial_time, balance_capital, balance_target, balance_source, balance_bnb)
        assert(balance_capital - balance_bnb > 100)

        # TODO: disable waiting here?
        # wait(time_delta, initial_time, True)

        while status['msg'] == 'Normal' and status['success'] == True:
            try:
                X, timeTo = get_recent_data(target_symbol, size = w + 1, type='h', aggregate=aggregate)

                if type == 'sma':
                    ma = np.diff(sma(X[:, 0] / X[0, 0], w))
                else:
                    alpha = 1 - 1 / w
                    ma = np.diff(ema(X[:, 0] / X[0, 0], alpha, 1.0))

                buy = ma[0] > 0
                sell = not buy

                print(ma[0])

                price = X[-1, 0]
                action = 'DO NOTHING'

                if buy and balance_source > 100:
                    success = buy_assets(client)
                    if success:
                        action = 'BUY'
                elif sell and balance_target > 100:
                    success = sell_assets(client)
                    if success:
                        action = 'SELL'


                if action != 'DO NOTHING':
                    print()
                    print(timeTo, action, price)
                    check_bnb(client)
                    balance_capital, balance_target, balance_source, balance_bnb = get_total_balance(client)
                    print(balance_capital, balance_target, balance_source, balance_bnb)
                    print()

                wait(time_delta, timeTo, False)
            except ConnectionError as e:
                print(e)
                status = get_status(client)
                print('Status:', status)
            except ReadTimeout as e:
                print(e)
                status = get_status(client)
                print('Status:', status)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Status:', get_status(client))
        print('Exiting trading pipeline...')

        cancel_orders(client, symbol)
        balance_capital, balance_target, balance_source, balance_bnb = get_total_balance(client)
        print()
        print(balance_capital, balance_target, balance_source, balance_bnb)

if __name__ == '__main__':
    trading_pipeline()
