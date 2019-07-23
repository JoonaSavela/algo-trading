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

        obj = {'target': 7.120549378179335, 'params': {'buy_threshold': 0.18044507894143255, 'change_threshold': 2.395989524989884, 'ha_threshold': 0.00022740452198337369, 'look_back_size': 11, 'maxlen': 4, 'min_threshold': 1.6516799323842664, 'rolling_min_window_size': 123, 'sell_threshold': 0.007327020114091631, 'stop_loss': -0.05763852099375464, 'take_profit': 0.016606080089034008, 'waiting_time': 0, 'window_size': 42, 'window_size1': 2 * 14 * 14, 'window_size2': 4 * 14 * 14}}
        stop_loss_take_profit = True
        restrictive = False

        params = obj['params']

        # print(params)

        strategy = Main_Strategy(params, stop_loss_take_profit, restrictive)

        # wealths = get_trades(count = 4)
        # trades = wealths[-3:] / wealths[:3]
        # print(trades)
        #
        # for trade in trades:
        #     deque_criterion.append(trade)
        #
        # print(deque_criterion.get_profit())

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

                X, timeTo = get_recent_data(symbol1, size = strategy.size())

                buy, sell = strategy.get_output(X, reset = False)

                # print(buy, sell)

                if buy:
                    balance_usdt = asset_balance(client, 'USDT')
                    high = binance_price(client, symbol)
                    # cancel_orders(client, symbol, 'SELL')
                    buy_assets(client, symbol, high, balance_usdt)
                    action = 'BUY'
                elif sell:
                    balance_symbol = asset_balance(client, symbol1)
                    # cancel_orders(client, symbol, 'BUY')
                    sell_assets(client, symbol, balance_symbol)
                    action = 'SELL'
                else:
                    action = 'DO NOTHING'

                price = X[-1, 0]

                print(timeTo, action, price)

                # if logic_criterion.recently_sold and not deque_criterion.buy(timeTo / 60):
                #     profit = deque_criterion.get_profit() - 1.0
                #     waiting_time = deque_criterion.get_waiting_time(profit) * 60
                #     print('Sleeping for', waiting_time // (60 * 60), 'hours')
                #     time.sleep(waiting_time)
                #     logic_criterion.recently_sold = False
                #     _, timeTo = get_recent_data(symbol1)

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
