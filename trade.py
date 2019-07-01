from keys import binance_api_key, binance_secret_key
# from model import build_model
import binance
from binance.client import Client
from binance.enums import *
import time
from data import coins, get_recent_data
import numpy as np
from utils import round_to_n, floor_to_n, stochastic_oscillator, heikin_ashi, sma, std
from math import floor, log10
import json
from strategy import Stochastic_criterion, Heikin_ashi_criterion, Bollinger_criterion, Stop_loss_criterion
from requests.exceptions import ConnectionError

def asset_balance(client, symbol):
    response = client.get_asset_balance(asset=symbol)
    time.sleep(0.05)
    # print(response)
    return float(response['free'])

def binance_price(client, symbol):
    return float(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)[-1][2])

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
        print(e)
        # pass

def buy_assets(client, symbol, prev_close, balance_usdt, amount = 1):
    quantity = floor_to_n(balance_usdt / prev_close * amount, 4)
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity)
        # print('Bought:', order)
        time.sleep(0.1)
    except binance.exceptions.BinanceAPIException as e:
        print(e)
        # pass


def trading_pipeline():
    client = Client(binance_api_key, binance_secret_key)
    # symbol = np.random.choice(coins) + 'USDT'
    symbol1 = 'ETH'
    symbol = symbol1 + 'USDT'
    print(symbol1)

    try:
        print('Starting trading pipeline...')

        status = client.get_account_status()
        print('Status:', status)
        check_bnb(client)

        _, initial_time = get_recent_data(symbol1)
        initial_capital = asset_balance(client, 'USDT')
        initial_bnb = asset_balance(client, 'BNB')
        print(initial_time, initial_capital, asset_balance(client, symbol1), initial_bnb)

        window_size = 1 * 14
        k = 1

        stochastic_criterion = Stochastic_criterion(0.02, 0.065)
        ha_criterion = Heikin_ashi_criterion()
        # bollinger_criterion = Bollinger_criterion(8)
        stop_loss = Stop_loss_criterion(-0.0075)

        time_diff = time.time() - initial_time
        waiting_time = 60 - time_diff
        print('waiting', waiting_time, 'seconds')
        time.sleep(waiting_time)

        while status['msg'] == 'Normal' and status['success'] == True:
            try:
                X, timeTo = get_recent_data(symbol1, size = window_size * 2 + 2 + k - 1)

                tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
                ma = sma(tp, window_size)

                X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)

                stochastic = stochastic_oscillator(X_corrected, window_size, k)
                ha = heikin_ashi(X)

                price = X[-1 ,0]

                if ha_criterion.buy(ha[-1, :]) and stochastic_criterion.buy(stochastic[-1]):
                    balance_usdt = asset_balance(client, 'USDT')
                    high = binance_price(client, symbol)
                    # cancel_orders(client, symbol, 'SELL')
                    buy_assets(client, symbol, high, balance_usdt)
                    action = 'BUY'
                elif ha_criterion.sell(ha[-1, :]) and \
                        (stochastic_criterion.sell(stochastic[-1]) or \
                        stop_loss.sell(price, X[-1, 3])):
                    balance_symbol = asset_balance(client, symbol1)
                    # cancel_orders(client, symbol, 'BUY')
                    sell_assets(client, symbol, balance_symbol)
                    action = 'SELL'
                else:
                    action = 'DO NOTHING'

                print()
                # print(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)[-1])
                print(timeTo, action, price, \
                    round_to_n(stochastic[-1], 3), round_to_n(ma[-1], 5))

                time.sleep(20)
                check_bnb(client)
                # open_orders = client.get_open_orders(symbol=symbol)
                balance_symbol = asset_balance(client, symbol1)
                balance_usdt = asset_balance(client, 'USDT')
                balance_bnb = asset_balance(client, 'BNB')
                print(balance_usdt, balance_symbol, balance_bnb)

                time_diff = time.time() - timeTo
                waiting_time = 60 - time_diff
                time.sleep(waiting_time)
            except ConnectionError as e:
                print(e)
                status = client.get_account_status()
                print('Status:', status)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        print('Status:', client.get_account_status())
        print('Exiting trading pipeline...')

        cancel_orders(client, symbol)
        balance_symbol = asset_balance(client, symbol1)
        if balance_symbol > 0:
            sell_assets(client, symbol, balance_symbol)
        time.sleep(0.1)

        _, final_time = get_recent_data(symbol1)
        final_capital = asset_balance(client, 'USDT')
        final_bnb = asset_balance(client, 'BNB')
        print(final_time, final_capital, asset_balance(client, symbol1), final_bnb)

        obj = {
            'symbol': symbol,
            'initial_time': str(initial_time),
            'final_time': str(final_time),
            'initial_capital': str(initial_capital),
            'final_capital': str(final_capital),
            'initial_bnb': str(initial_bnb),
            'final_bnb': str(final_bnb)
        }
        filename = 'trading_logs/' + symbol1 + '-' + str(initial_time) + '-' + str(final_time) + '.json'

        if not initial_capital == final_capital:
            with open(filename, 'w') as file:
                json.dump(obj, file)


if __name__ == '__main__':
    trading_pipeline()
