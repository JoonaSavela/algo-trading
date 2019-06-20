from keys import binance_api_key, binance_secret_key
from model import build_model
import binance
from binance.client import Client
from binance.enums import *
import time
from data import coins, get_recent_data
import numpy as np
from utils import floor_to_n
from math import floor, log10
import json

weights_filename = 'models/model_weights.pt'
input_length = 14 * 4

def asset_balance(client, symbol):
    return float(client.get_asset_balance(asset=symbol)['free'])

def check_bnb(client):
    balance_bnb = asset_balance(client, 'BNB')
    if balance_bnb < 0.2:
        try:
            client.order_market_buy(symbol='BNBUSDT', quantity=0.5)
            print('Bought BNB')
            time.sleep(0.1)
        except:
            pass

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

def sell_assets(client, symbol, balance_symbol, amount = 1):
    quantity = floor_to_n(balance_symbol * amount, 4)
    try:
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity)
        # print('Sold:', order)
        time.sleep(0.1)
    except binance.exceptions.BinanceAPIException as e:
        # print(e)
        pass

def buy_assets(client, symbol, prev_close, balance_usdt, amount):
    quantity = floor_to_n(balance_usdt / prev_close * amount, 4)
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity)
        # print('Bought:', order)
        time.sleep(0.1)
    except:
        pass


def trading_pipeline():
    model = build_model()
    model.load_weights(weights_filename)

    client = Client(binance_api_key, binance_secret_key)
    # symbol = np.random.choice(coins) + 'USDT'
    symbol = 'BTCUSDT'
    symbol1 = symbol[:3] # TODO: make this the other way around
    print(symbol1)

    try:
        print()
        print('Starting trading pipeline...')

        # balance_symbol = asset_balance(client, symbol1)
        # sell_assets(client, symbol, balance_symbol)
        check_bnb(client)
        # open_orders = client.get_open_orders(symbol=symbol)
        # if len(open_orders) > 0:
        #     print(open_orders)

        _, initial_time = get_recent_data(symbol1)
        initial_capital = asset_balance(client, 'USDT')
        print(initial_time, initial_capital, asset_balance(client, symbol1))

        time_diff = time.time() - initial_time
        waiting_time = 60 - time_diff
        print('waiting', waiting_time, 'seconds')
        time.sleep(waiting_time)

        while True:
            X, timeTo = get_recent_data(symbol1)
            balance_symbol = asset_balance(client, symbol1)
            balance_usdt = asset_balance(client, 'USDT')

            means = np.reshape(np.mean(X, axis=0), (1,6))
            stds = np.reshape(np.std(X, axis=0), (1,6))
            inp = np.reshape((X - means) / stds, (1, input_length, 6))
            BUY, SELL, DO_NOTHING, amount = tuple(np.reshape(model.predict(inp), (4,)))

            if BUY > SELL and BUY > DO_NOTHING:
                cancel_orders(client, symbol, 'SELL')
                buy_assets(client, symbol, X[-1, 0], balance_usdt, amount)
                print('BUY', amount)
            elif SELL > BUY and SELL > DO_NOTHING:
                cancel_orders(client, symbol, 'BUY')
                sell_assets(client, symbol, balance_symbol, amount)
                print('SELL', amount)
            else:
                print()

            print(timeTo, X[-1, 0], balance_symbol, balance_usdt)

            time.sleep(20)
            check_bnb(client)
            open_orders = client.get_open_orders(symbol=symbol)
            balance_symbol = asset_balance(client, symbol1)
            balance_usdt = asset_balance(client, 'USDT')
            print(balance_symbol, balance_usdt, len(open_orders))

            time_diff = time.time() - timeTo
            waiting_time = 60 - time_diff
            time.sleep(waiting_time)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Exiting trading pipeline...')

        cancel_orders(client, symbol)
        balance_symbol = asset_balance(client, symbol1)
        sell_assets(client, symbol, balance_symbol)
        time.sleep(0.1)

        _, final_time = get_recent_data(symbol1)
        final_capital = asset_balance(client, 'USDT')
        print(final_time, final_capital, asset_balance(client, symbol1))

        obj = {
            'symbol': symbol,
            'initial_time': str(initial_time),
            'final_time': str(final_time),
            'initial_capital': str(initial_capital),
            'final_capital': str(final_capital)
        }
        filename = 'trading_logs/' + str(initial_time) + '-' + str(final_time) + '.json'

        with open(filename, 'w') as file:
            json.dump(obj, file)


if __name__ == '__main__':
    trading_pipeline()
