from keys import ftx_api_key, ftx_secret_key, email_address, email_password
# from model import build_model
from ftx.rest.client import FtxClient
import time
from datetime import datetime
from data import get_recent_data
import numpy as np
import pandas as pd
from utils import aggregate as aggregate_F
from utils import floor_to_n, print_dict, get_gcd, get_lcm, round_to_n, \
    choose_get_buys_and_sells_fn, get_filtered_strategies_and_weights, \
    get_parameter_names
import json
import smtplib
from functools import reduce
import os
from math import isnan

source_symbol = 'USD'
min_amount = 0.001

def get_symbol(coin, m, bear = False):
    res = coin
    if m > 1:
        if coin == 'BTC':
            res = ''
        res += 'BULL' if not bear else 'BEAR'

    return res

def get_symbol_from_key(strategy_key, bear = False):
    coin = strategy_key.split('_')[0]
    m, m_bear = tuple([int(x) for x in strategy_key.split('_')[3:]])

    return get_symbol(coin, m if not bear else m_bear, bear=bear)


# TODO: cleaner way? recall pandas usage, e.g. Data science handbook/cookbook
def ftx_price(client, symbol, side = 'ask'):
    markets = pd.DataFrame(client.list_markets())
    res = float(markets[(markets['baseCurrency'] == symbol) & (markets['quoteCurrency'] == source_symbol)][side])
    time.sleep(0.05)
    return res

# TODO: remove since not used
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


def get_total_balance(client, separate = True, filter = None):
    balances = pd.DataFrame(client.get_balances()).set_index('coin')
    time.sleep(0.05)
    if filter is not None:
        li = [ix in filter for ix in balances.index]
        balances = balances.loc[li, :]
    total_balance = balances['usdValue'].sum()

    if separate:
        return total_balance, balances

    return total_balance

def sell_assets(client, symbol, amount, round_n = 4, debug = False):
    quantity = floor_to_n(amount, round_n)

    if not debug:
        client.place_order(
            market = symbol + '/' +  source_symbol,
            side = 'sell',
            price = None,
            size = quantity,
            type = 'market'
        )
        time.sleep(0.1)

    return quantity

def buy_assets(client, symbol, usd_amount, round_n = 4, debug = False):
    price = ftx_price(client, symbol, side = 'ask')
    quantity = floor_to_n(usd_amount / price, round_n)

    if not debug:
        client.place_order(
            market = symbol + '/' +  source_symbol,
            side = 'buy',
            price = None,
            size = quantity,
            type = 'market'
        )
        time.sleep(0.1)

    return quantity

def place_trigger_order(client, symbol, buy_price, amount, trigger_name, trigger_param, round_n = 4, debug = False):
    quantity = floor_to_n(amount, round_n)

    if not debug:
        if trigger_name == 'stop_loss':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = quantity,
                order_type = 'stop',
                trigger_price = buy_price * trigger_param
            )
        elif trigger_name == 'take_profit':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = quantity,
                order_type = 'takeProfit',
                trigger_price = buy_price * trigger_param
            )
        elif trigger_name == 'trailing':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = quantity,
                order_type = 'trailingStop',
                trail_value = -(1 - trigger_param) * buy_price
            )
        else:
            raise ValueError('trigger_name')

        time.sleep(0.1)
    else:
        res = {
            'id': None
        }

    return res['id'], quantity



def wait(time_delta, initial_time, verbose = False):
    target_time = initial_time + time_delta
    waiting_time = target_time - time.time()
    while waiting_time < 0:
        waiting_time += 60 * 60
    if verbose:
        print()
        if waiting_time / (60 * 60) < 1:
            print('waiting', round_to_n(waiting_time / 60, 4), 'minutes')
        else:
            print('waiting', round_to_n(waiting_time / (60 * 60), 4), 'hours')
    time.sleep(waiting_time)

def cancel_orders(client):
    client.cancel_orders()
    time.sleep(0.05)


# TODO: implement
def cancel_orders_and_sell_all(client):
    cancel_orders(client)

    total_balance, balances = get_total_balance(client, separate = True)


def get_conditional_orders(client):
    res = {}
    for cond_order in client.get_conditional_orders():
        res[cond_order['id']] = cond_order
    time.sleep(0.05)
    return res


def send_error_email(address, password):
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        smtp.login(address, password)

        subject = 'Error in trading algorithm'

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        body = f'The error occurred at {dt_string}'

        msg = f'Subject: {subject}\n\n{body}'

        smtp.sendmail(address, address, msg)



# buy_info: DataFrame of relevant information
#   - buy_state: dtypes = [None, bool]
#   - buy_price: dtypes = [None, float]
#   - trigger_name: dtypes = [None, str]
#   - trigger_param: dtypes = [None, float]
#   - weight: dtypes = [None, float]
#   - trigger_order_id: dtypes = [None, int]
def balance_portfolio(client, buy_info, debug = False):
    open_trigger_orders = get_conditional_orders(client)

    triggered = {}

    # detect triggered trigger orders
    for strategy_key in buy_info.columns:
        if buy_info[strategy_key]['trigger_order_id'] is None or \
                isnan(buy_info[strategy_key]['trigger_order_id']):
            triggered[strategy_key] = False
        else:
            triggered[strategy_key] = \
                buy_info[strategy_key]['trigger_order_id'] not in open_trigger_orders or \
                open_trigger_orders[buy_info[strategy_key]['trigger_order_id']]['status'] != 'open'

    # cancel orders
    if not debug:
        cancel_orders(client)

    symbols = {}
    weights = {
        source_symbol: 0
    }

    # Get all possible symbols
    for strategy_key in buy_info.columns:
        symbol = get_symbol_from_key(strategy_key, bear = False)
        if symbol not in weights:
            weights[symbol] = 0

        symbol = get_symbol_from_key(strategy_key, bear = True)
        if symbol not in weights:
            weights[symbol] = 0

    # Get the target weights for those symbols
    for strategy_key in buy_info.columns:
        # if trigger order triggered, symbol = source_symbol
        if triggered[strategy_key]:
            symbol = source_symbol
        elif buy_info[strategy_key]['buy_state'] == True:
            symbol = get_symbol_from_key(strategy_key, bear = False)
        elif buy_info[strategy_key]['buy_state'] == False:
            symbol = get_symbol_from_key(strategy_key, bear = True)
        else:
            symbol = source_symbol

        symbols[strategy_key] = symbol

        weights[symbol] += buy_info[strategy_key]['weight']

    symbols = pd.Series(symbols)
    target_weights = pd.Series(weights)

    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    actual_weights = balances['usdValue'] / total_balance
    weight_diffs = target_weights - actual_weights

    # sell negatives
    li_neg = weight_diffs < 0

    for symbol, weight_diff in weight_diffs[li_neg].items():
        if symbol != source_symbol and np.abs(weight_diff) > min_amount:
            amount = np.abs(weight_diff / actual_weights[symbol]) * balances['total'][symbol]
            quantity = sell_assets(client, symbol, amount, round_n = 5, debug = debug)


    if not debug:
        time.sleep(1)

    # buy positives
    li_pos = ~li_neg

    for symbol, weight_diff in weight_diffs[li_pos].items():
        if symbol != source_symbol and weight_diff > min_amount:
            usd_amount = weight_diff * total_balance
            quantity = buy_assets(client, symbol, usd_amount, round_n = 5, debug = debug)

    if not debug:
        time.sleep(1)

    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    quantities = {}

    # (re)apply trigger orders
    for strategy_key, symbol in symbols.items():
        if symbol != source_symbol:
            amount = buy_info[strategy_key]['weight'] / target_weights[symbol] * balances['total'][symbol]

            id, quantity = place_trigger_order(
                client,
                symbol,
                buy_info[strategy_key]['buy_price'],
                amount,
                buy_info[strategy_key]['trigger_name'],
                buy_info[strategy_key]['trigger_param'],
                round_n = 5,
                debug = debug
            )

            buy_info[strategy_key]['trigger_order_id'] = id
            quantities[strategy_key] = quantity
        else:
            quantities[strategy_key] = None

    # print all that was done
    print_df = pd.DataFrame({
        'buy_state': buy_info.loc['buy_state'],
        'buy_price': buy_info.loc['buy_price'],
        'triggered': triggered,
        'symbol': symbols,
        'trigger_quantity': quantities,
        'trigger_order_id': buy_info.loc['trigger_order_id'],
    }).transpose()
    print(print_df)
    print()

    return buy_info


# TODO: have multiple counters for low freq strategies with same aggregate_N
def trading_pipeline(
    buy_info = None,
    coins = ['ETH', 'BTC'],
    freqs = ['low', 'high'],
    strategy_types = ['ma', 'macross'],
    ms = [1, 3],
    m_bears = [0, 3],
    ask_for_input = True
):
    print('Starting trading pipeline...')
    print()

    client = FtxClient(ftx_api_key, ftx_secret_key)

    debug = False

    if ask_for_input:
        debug = input('Is this a debug run? ')
        if 'y' in debug:
            debug = True

        if debug:
            print('Debug flag set')

    strategies, weights = get_filtered_strategies_and_weights(
        coins,
        freqs,
        strategy_types,
        ms,
        m_bears
    )
    get_buys_and_sells_fns = {}
    N_parameters = {}

    for strategy_key in strategies.keys():
        strategy_type = strategy_key.split('_')[2]

        N_parameters[strategy_key] = len(get_parameter_names(strategy_type))
        get_buys_and_sells_fns[strategy_key] = choose_get_buys_and_sells_fn(strategy_type)


    if os.path.exists('optim_results/displacements.json'):
        with open('optim_results/displacements.json', 'r') as file:
            displacements = json.load(file)

    for k in strategies.keys():
        if k not in displacements:
            displacements[k] = 0
        else:
            displacements[k] = displacements[k][0]

    displacement_values = np.sort(np.array([v for v in displacements.values()]))

    strategy_keys = np.array(sorted(strategies.keys(), key = lambda x: displacements[x]))

    aggregate_Ns_all = np.array([strategies[key]['params'][0] for key in strategy_keys])
    lcm = reduce(lambda x, y: get_lcm(x, y), aggregate_Ns_all)
    gcd = reduce(lambda x, y: get_gcd(x, y), aggregate_Ns_all)
    counter = 0

    if buy_info is None:
        # DataFrame of relevant information
        #   - buy_state: dtypes = [None, bool]
        #   - buy_price: dtypes = [None, float]
        #   - trigger_name: dtypes = [None, str]
        #   - trigger_param: dtypes = [None, float]
        #   - weight: dtypes = [None, float]
        #   - trigger_order_id: dtypes = [None, int]
        buy_info = pd.DataFrame(
            index=['buy_state', 'buy_price', 'trigger_name', 'trigger_param', 'weight', 'trigger_order_id'],
            columns=strategy_keys
        )
        for key in strategy_keys:
            buy_info[key]['weight'] = weights[key]

    error_flag = True

    seconds = time.time()
    localtime = time.localtime(seconds)
    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
    timeTo0 = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
    if not debug:
        wait(0, timeTo0, verbose = False)

    try:
        while True:
            seconds = time.time()
            localtime = time.localtime(seconds)
            now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
            timeTo0 = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())

            li_counter = (counter % aggregate_Ns_all) == 0
            displacement_values_unique = np.unique(displacement_values[li_counter])

            for displacement in displacement_values_unique:
                if displacement > 0:
                    seconds = time.time() + (59 - displacement) * 60
                    localtime = time.localtime(seconds)
                    now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
                    timeTo = time.mktime(datetime.strptime(now_string[:-6], "%d/%m/%Y %H").timetuple())
                    if not debug:
                        wait(displacement * 60, timeTo, verbose = debug)

                # if debug:
                #     if displacement == 0:
                #         timeTo = timeTo0
                #     localtime = time.localtime(timeTo)
                #     now_string = time.strftime("%d/%m/%Y %H:%M:%S", localtime)
                #     print(displacement, now_string)

                li = displacement_values[li_counter] == displacement
                keys = strategy_keys[li_counter][li]

                aggregate_Ns = np.array([strategies[key]['params'][0] for key in keys])
                ws = np.array([strategies[key]['params'][1] for key in keys])

                sizes = (ws + 1) * aggregate_Ns
                if displacement > 0:
                    sizes *= 60

                coins = np.unique([key.split('_')[0] for key in keys])

                X_dict = {}

                for coin in coins:
                    if displacement > 0:
                        X, _ = get_recent_data(coin, size = np.max(sizes), type='m')
                    else:
                        X, _ = get_recent_data(coin, size = np.max(sizes), type='h')
                    if coin not in X_dict:
                        X_dict[coin] = X

                changed = False

                for key in keys:
                    coin = key.split('_')[0]
                    m, m_bear = tuple([int(x) for x in key.split('_')[3:]])

                    args = strategies[key]['params'][:N_parameters[key]]
                    sltp_args = strategies[key]['params'][N_parameters[key]:]
                    aggregate_N = args[0]

                    X = aggregate_F(X_dict[coin], aggregate_N, from_minute = displacement > 0)

                    w = args[1]

                    buy, sell, N = get_buys_and_sells_fns[key](
                        X[-(w + 1):, :],
                        1,
                        *args[1:],
                        as_boolean=True,
                        from_minute = False
                    )

                    assert(N == 1)

                    if buy:
                        if buy != buy_info[key]['buy_state']:
                            buy_info[key]['buy_state'] = True
                            buy_info[key]['buy_price'] = ftx_price(client, get_symbol(coin, m, bear = False))
                            buy_info[key]['trigger_name'] = sltp_args[0]
                            buy_info[key]['trigger_param'] = sltp_args[1]
                            buy_info[key]['trigger_order_id'] = None
                            changed = True

                    elif sell and len(sltp_args) == 4:
                        if (not sell) != buy_info[key]['buy_state']:
                            buy_info[key]['buy_state'] = False
                            buy_info[key]['buy_price'] = ftx_price(client, get_symbol(coin, m_bear, bear = True))
                            buy_info[key]['trigger_name'] = sltp_args[2]
                            buy_info[key]['trigger_param'] = sltp_args[3]
                            buy_info[key]['trigger_order_id'] = None
                            changed = True

                    elif sell and len(sltp_args) == 2:
                        if buy_info[key]['buy_state'] == True:
                            buy_info[key]['buy_state'] = None
                            buy_info[key]['buy_price'] = None
                            buy_info[key]['trigger_name'] = None
                            buy_info[key]['trigger_param'] = None
                            buy_info[key]['trigger_order_id'] = None
                            changed = True

                if debug:
                    print(buy_info)
                    print()

                if changed:
                    buy_info = balance_portfolio(client, buy_info, debug = debug)

            counter = (counter + gcd) % lcm

            wait(60 * 60 * gcd, timeTo0, verbose = debug)

            if debug:
                break
    except KeyboardInterrupt:
        error_flag = False
    finally:
        print('Exiting trading pipeline...')
        print()

        if error_flag:
            send_error_email(email_address, email_password)

        # return error_flag, False, buy_info


    return error_flag, False, buy_info



if __name__ == '__main__':
    buy_info = None
    error_flag = True
    ask_for_input = True

    while error_flag:
        error_flag, ask_for_input, buy_info = trading_pipeline(
            buy_info = buy_info,
            ask_for_input = ask_for_input
        )
        # time.sleep(60)
