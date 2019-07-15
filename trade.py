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
from strategy import *
from requests.exceptions import ConnectionError, ReadTimeout
from visualize import get_trades

def asset_balance(client, symbol):
    response = client.get_asset_balance(asset=symbol)
    time.sleep(0.05)
    # print(response)
    return float(response['free'])

def binance_price(client, symbol):
    return float(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)[-1][2])

def check_bnb(client):
    balance_bnb = asset_balance(client, 'BNB')
    if balance_bnb < 0.25:
        try:
            client.order_market_buy(symbol='BNBUSDT', quantity=0.5)
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

        window_size1 = 3 * 14
        window_size2 = 1 * 14
        window_size3 = 1 * 14
        k = 1

        stochastic_criterion = Stochastic_criterion(0.04, 0.08)
        ha_criterion = Heikin_ashi_criterion()
        stop_loss = Stop_loss_criterion(-0.03)
        take_profit = Take_profit_criterion(0.01)
        trend_criterion = Trend_criterion(0.02)
        deque_criterion = Deque_criterion(3, 10 * 60)
        # deque_criterion.sell_time = initial_time / 60

        wealths = get_trades(count = 4)
        trades = wealths[-3:] / wealths[:3]
        print(trades)

        for trade in trades:
            deque_criterion.append(trade)

        print(deque_criterion.get_profit())

        time_diff = time.time() - initial_time
        waiting_time = 60 - time_diff
        while waiting_time < 0:
            waiting_time += 60
        print('waiting', waiting_time, 'seconds')
        time.sleep(waiting_time)

        while status['msg'] == 'Normal' and status['success'] == True:
            try:
                X, timeTo = get_recent_data(symbol1, size = window_size1 + np.max([window_size3, window_size2]) + 2 + k - 1)

                tp = np.mean(X[:, :3], axis = 1).reshape((X.shape[0], 1))
                ma = sma(tp, window_size1)

                X_corrected = X[-ma.shape[0]:, :4] - np.repeat(ma.reshape((-1, 1)), 4, axis = 1)

                stochastic = stochastic_oscillator(X_corrected, window_size2, k)
                ha = heikin_ashi(X)

                X_corrected /= np.repeat(X[-X_corrected.shape[0]:, 0].reshape((-1, 1)), 4, axis = 1)

                ma_corrected = sma(X_corrected, window_size3)

                price = X[-1 ,0]
                stoch = stochastic[-1]

                if ha_criterion.buy(ha[-1, :]) and take_profit.buy() and \
                        stop_loss.buy() and deque_criterion.buy(timeTo / 60) and \
                        (stochastic_criterion.buy(stoch) or \
                        trend_criterion.buy(ma_corrected[-1])):
                    take_profit.buy_price = price
                    stop_loss.buy_price = price
                    deque_criterion.sell_time = None
                    balance_usdt = asset_balance(client, 'USDT')
                    high = binance_price(client, symbol)
                    # cancel_orders(client, symbol, 'SELL')
                    buy_assets(client, symbol, high, balance_usdt)
                    action = 'BUY'
                elif ha_criterion.sell(ha[-1, :]) and \
                        (stochastic_criterion.sell(stoch) or \
                        trend_criterion.sell(ma_corrected[-1]) or \
                        stop_loss.sell(price) or \
                        take_profit.sell(price)):
                    if take_profit.buy_price is not None:
                        trade = price / take_profit.buy_price
                        deque_criterion.append(trade)
                    take_profit.buy_price = None
                    stop_loss.buy_price = None
                    deque_criterion.sell_time = timeTo / 60
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

                if deque_criterion.sell_time is not None and not deque_criterion.buy(timeTo / 60):
                    waiting_time = deque_criterion.waiting_time * 60
                    print('Sleeping for', waiting_time // (60 * 60), 'hours')
                    time.sleep(waiting_time)
                    _, timeTo = get_recent_data(symbol1)

                time.sleep(20)
                check_bnb(client)
                # open_orders = client.get_open_orders(symbol=symbol)
                balance_symbol = asset_balance(client, symbol1)
                balance_usdt = asset_balance(client, 'USDT')
                balance_bnb = asset_balance(client, 'BNB')
                print(balance_usdt, balance_symbol, balance_bnb)

                time_diff = time.time() - timeTo
                waiting_time = 60 - time_diff
                while waiting_time < 0:
                    waiting_time += 60
                time.sleep(waiting_time)
            except ConnectionError as e:
                print(e)
                status = client.get_account_status()
                print('Status:', status)
            except ReadTimeout as e:
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

        # if not initial_capital == final_capital:
        #     with open(filename, 'w') as file:
        #         json.dump(obj, file)


if __name__ == '__main__':
    trading_pipeline()
