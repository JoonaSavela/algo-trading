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
from parameters import parameters

def asset_balance(client, symbol):
    response = client.get_asset_balance(asset=symbol)
    time.sleep(0.05)
    return float(response['free'])

def binance_price(client, symbol):
    return float(client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=10)[-1][2])

def check_bnb(client):
    balance_usdt = asset_balance(client, 'USDT')
    time.sleep(0.05)
    if balance_usdt > 50:
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
        time.sleep(0.1)
        return True
    except binance.exceptions.BinanceAPIException as e:
        print(e)
        return False

def buy_assets(client, symbol, prev_close, balance_usdt, amount = 1):
    quantity = floor_to_n(balance_usdt / prev_close * amount, 4)
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity)
        time.sleep(0.1)
        return True
    except binance.exceptions.BinanceAPIException as e:
        print(e)
        return False


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

        stop_loss_take_profit = True
        restrictive = True

        obj = parameters[2]

        params = obj['params']

        if 'decay' not in params:
            params['decay'] = 0.0001

        strategy = Main_Strategy(params, stop_loss_take_profit, restrictive)
        X_size = strategy.size()

        wealths = get_trades(count = 4)
        trades = wealths[-3:] / wealths[:3]
        print(trades)

        # for trade in trades:
        #     strategy.deque_criterion.append(trade)
        #
        # print(strategy.deque_criterion.get_profit())

        time_diff = time.time() - initial_time
        waiting_time = 60 - time_diff
        while waiting_time < 0:
            waiting_time += 60
        print()
        print('waiting', waiting_time, 'seconds')
        time.sleep(waiting_time)

        while status['msg'] == 'Normal' and status['success'] == True:
            try:
                print()

                X, timeTo = get_recent_data(symbol1, size = X_size)

                buy, sell, buy_and_criteria, sell_and_criteria, buy_or_criteria, sell_or_criteria = \
                    strategy.get_output(X, timeTo / 60, reset = False, update = False)

                # print(buy, sell)

                price = X[-1, 0]
                action = 'DO NOTHING'

                if buy:
                    balance_usdt = asset_balance(client, 'USDT')
                    high = binance_price(client, symbol)

                    success = buy_assets(client, symbol, high, balance_usdt)

                    if success:
                        strategy.update_after_buy(price, timeTo / 60)
                        action = 'BUY'
                elif sell:
                    balance_symbol = asset_balance(client, symbol1)

                    success = sell_assets(client, symbol, balance_symbol)

                    if success:
                        strategy.update_after_sell(price, timeTo / 60)
                        action = 'SELL'

                        print(strategy.deque_criterion.trades)
                        print(strategy.deque_criterion.get_profit())

                if action == 'BUY':
                    print(timeTo, action, price, buy_and_criteria, buy_or_criteria)
                elif action == 'SELL':
                    print(timeTo, action, price, sell_and_criteria, sell_or_criteria)

                if restrictive and strategy.logic_criterion.recently_sold and \
                        not strategy.deque_criterion.buy({'current_time': timeTo / 60}):
                    profit = strategy.deque_criterion.get_profit() - 1.0
                    waiting_time = strategy.deque_criterion.get_waiting_time(profit) * 60
                    print('Sleeping for', waiting_time // (60 * 60), 'hours')
                    time.sleep(waiting_time)
                    strategy.logic_criterion.recently_sold = False
                    _, timeTo = get_recent_data(symbol1)

                time.sleep(20)
                if action != 'DO NOTHING':
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
