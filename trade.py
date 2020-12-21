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
# from math import isnan

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

def sell_assets(client, symbol, size, round_n = 4, debug = False):
    size = floor_to_n(size, round_n)

    if not debug:
        client.place_order(
            market = symbol + '/' +  source_symbol,
            side = 'sell',
            price = None,
            size = size,
            type = 'market'
        )
        time.sleep(0.1)

    return size

def buy_assets(client, symbol, usd_size, round_n = 4, debug = False):
    price = ftx_price(client, symbol, side = 'ask')
    size = floor_to_n(usd_size / price, round_n)

    if not debug:
        client.place_order(
            market = symbol + '/' +  source_symbol,
            side = 'buy',
            price = None,
            size = size,
            type = 'market'
        )
        time.sleep(0.1)

    return size

def place_trigger_order(client, symbol, buy_price, size, trigger_name, trigger_param, round_n = 4, debug = False):
    size = floor_to_n(size, round_n)

    if not debug:
        if trigger_name == 'stop_loss':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = size,
                order_type = 'stop',
                triggerPrice = buy_price * trigger_param
            )
        elif trigger_name == 'take_profit':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = size,
                order_type = 'takeProfit',
                triggerPrice = buy_price * trigger_param
            )
        elif trigger_name == 'trailing':
            res = client.place_conditional_order(
                market = symbol + '/' +  source_symbol,
                side = 'sell',
                size = size,
                order_type = 'trailingStop',
                trailValue = -(1 - trigger_param) * buy_price
            )
        else:
            raise ValueError('trigger_name')

        time.sleep(0.1)
    else:
        res = {
            'id': None
        }

    return res['id'], size

def modify_trigger_order_size(
    client,
    order_id,
    trigger_name,
    size,
    trail_value = None,
    round_n = 4,
    debug = False):

    size = floor_to_n(size, round_n)

    trigger_name_to_order_type = {
        'stop_loss': 'stop',
        'take_profit': 'takeProfit',
        'trailing': 'trailingStop'
    }

    if not debug:
        res = client.modify_conditional_order(
            order_id = order_id,
            order_type = trigger_name_to_order_type[trigger_name],
            size = size,
            trailValue = trail_value
        )
        time.sleep(0.1)
    else:
        res = {
            'id': None
        }

    return res['id'], size



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

def cancel_orders(client, debug = False):
    if not debug:
        client.cancel_orders()
        time.sleep(0.05)

def cancel_conditional_order(client, order_id, debug = False):
    if not debug:
        client.cancel_conditional_order(order_id)
        time.sleep(0.05)


def cancel_order(client, order_id, debug = False):
    if not debug:
        client.cancel_order(order_id)
        time.sleep(0.05)



# TODO: implement
def cancel_orders_and_sell_all(client):
    cancel_orders(client)

    total_balance, balances = get_total_balance(client, separate = True)


def get_conditional_orders(client):
    res = {}
    for cond_order in client.get_conditional_orders():
        res[str(cond_order['id'])] = cond_order
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
def balance_portfolio(client, buy_info, debug = False, verbose = False):
    open_trigger_orders = get_conditional_orders(client)

    triggered = {}
    trigger_order_ids = {}
    trailing_trigger_prices = {}

    # detect triggered trigger orders
    for strategy_key in buy_info.index:
        if pd.isna(buy_info.loc[strategy_key, 'trigger_order_id']):
            trigger_order_id = None
        else:
            trigger_order_id = str(int(buy_info.loc[strategy_key, 'trigger_order_id']))

        trigger_order_ids[strategy_key] = trigger_order_id

        if trigger_order_id is None:
            triggered[strategy_key] = False
        else:
            triggered[strategy_key] = \
                trigger_order_id not in open_trigger_orders or \
                open_trigger_orders[trigger_order_id]['status'] != 'open'

            if buy_info.loc[strategy_key, 'trigger_name'] == 'trailing':
                trailing_trigger_prices[strategy_key] = open_trigger_orders[trigger_order_id]['triggerPrice']

        if triggered[strategy_key]:
            trigger_order_ids[strategy_key] = None

    if verbose:
        print(trigger_order_ids)
        print(trailing_trigger_prices)

    symbols = {}
    weights = {
        source_symbol: 0
    }

    # Get all possible symbols
    for strategy_key in buy_info.index:
        symbol = get_symbol_from_key(strategy_key, bear = False)
        if symbol not in weights:
            weights[symbol] = 0

        symbol = get_symbol_from_key(strategy_key, bear = True)
        if symbol not in weights:
            weights[symbol] = 0

    # Get the target weights for those symbols
    for strategy_key in buy_info.index:
        # if trigger order triggered, symbol = source_symbol
        if triggered[strategy_key]:
            symbol = source_symbol
        elif buy_info.loc[strategy_key, 'buy_state'] == True:
            symbol = get_symbol_from_key(strategy_key, bear = False)
        elif buy_info.loc[strategy_key, 'buy_state'] == False:
            symbol = get_symbol_from_key(strategy_key, bear = True)
        else:
            symbol = source_symbol

        symbols[strategy_key] = symbol

        weights[symbol] += buy_info.loc[strategy_key, 'weight']

    if verbose:
        print(weights)

    symbols = pd.Series(symbols)
    target_weights = pd.Series(weights)

    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    actual_weights = balances['usdValue'] / total_balance
    weight_diffs = target_weights - actual_weights

    if verbose:
        print(weight_diffs)

    # modify trailing orders s.t. its weight is correct
    for strategy_key, symbol in symbols.items():
        if symbol != source_symbol and trigger_order_ids[strategy_key] is not None:
            if buy_info.loc[strategy_key, 'trigger_name'] == 'trailing':
                size = min(
                    1.0,
                    buy_info.loc[strategy_key, 'weight'] / actual_weights[symbol]
                )

                if size * balances['usdValue'][symbol] / total_balance > min_amount:
                    size *= balances['total'][symbol]

                    trail_value = trailing_trigger_prices[strategy_key] \
                        - ftx_price(client, symbol)

                    id, size = modify_trigger_order_size(
                        client,
                        trigger_order_ids[strategy_key],
                        buy_info.loc[strategy_key, 'trigger_name'],
                        size,
                        trail_value = trail_value,
                        round_n = 5,
                        debug = debug
                    )

                    id = str(int(id)) if id is not None else id
                    buy_info.loc[strategy_key, 'trigger_order_id'] = id
                    trigger_order_ids[strategy_key] = id

                    if verbose:
                        print(strategy_key, id, type(id))
                        print(f'Modified order, size = {size}')

    # cancel orders if if their type is not 'trailing'
    for cond_order_id, cond_order in open_trigger_orders.items():
        if cond_order['type'] != 'trailing_stop':
            cancel_conditional_order(client, cond_order_id, debug = debug)
            for strategy_key, trigger_order_id in trigger_order_ids.items():
                if trigger_order_id == cond_order_id:
                    buy_info.loc[strategy_key, 'trigger_order_id'] = np.nan
                    trigger_order_ids[strategy_key] = None
            if verbose:
                print(f'cancel order {cond_order_id}')

    if verbose:
        print(buy_info)

    # sell negatives
    li_neg = weight_diffs < 0

    for symbol, weight_diff in weight_diffs[li_neg].items():
        if symbol != source_symbol and np.abs(weight_diff) > min_amount:
            size = np.abs(weight_diff / actual_weights[symbol]) * balances['total'][symbol]
            size = sell_assets(client, symbol, size, round_n = 5, debug = debug)


    if not debug:
        time.sleep(1)

    # buy positives
    li_pos = ~li_neg

    for symbol, weight_diff in weight_diffs[li_pos].items():
        if symbol != source_symbol and weight_diff > min_amount:
            usd_size = weight_diff * total_balance
            size = buy_assets(client, symbol, usd_size, round_n = 5, debug = debug)

    if not debug:
        time.sleep(1)

    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    sizes = {}

    # (re)apply trigger orders if trigger_name is not 'trailing' or if trigger_order_id is None
    # else modify order s.t. its weight is correct
    for strategy_key, symbol in symbols.items():
        if symbol != source_symbol and not triggered[strategy_key]:
            size = buy_info.loc[strategy_key, 'weight'] / target_weights[symbol] * balances['total'][symbol]

            if buy_info.loc[strategy_key, 'trigger_name'] != 'trailing' or \
                    trigger_order_ids[strategy_key] is None:
                id, size = place_trigger_order(
                    client,
                    symbol,
                    buy_info.loc[strategy_key, 'buy_price'],
                    size,
                    buy_info.loc[strategy_key, 'trigger_name'],
                    buy_info.loc[strategy_key, 'trigger_param'],
                    round_n = 5,
                    debug = debug
                )
            else:
                trail_value = trailing_trigger_prices[strategy_key] \
                    - ftx_price(client, symbol)

                id, size = modify_trigger_order_size(
                    client,
                    trigger_order_ids[strategy_key],
                    buy_info.loc[strategy_key, 'trigger_name'],
                    size,
                    trail_value = trail_value,
                    round_n = 5,
                    debug = debug
                )

            id = str(int(id)) if id is not None else id
            buy_info.loc[strategy_key, 'trigger_order_id'] = id
            trigger_order_ids[strategy_key] = id

            sizes[strategy_key] = size

            if verbose:
                print(id, type(id))
                print(f'Modified or placed order, size = {size}')

        else:
            sizes[strategy_key] = None

    # print all that was done
    print_df = pd.DataFrame({
        'buy_state': buy_info['buy_state'],
        'buy_price': buy_info['buy_price'],
        'symbol': symbols,
        'triggered': triggered,
        'trigger_name': buy_info['trigger_name'],
        'trigger_size': sizes,
        'trigger_order_id': buy_info['trigger_order_id'],
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
    buy_info_from_file = True
    buy_info_file = 'optim_results/buy_info.csv'
    wait_for_signal_to_change = False

    if ask_for_input:
        debug_inp = input('Is this a debug run (y/[n])? ')
        if 'y' in debug_inp:
            debug = True

        if debug:
            print('Debug flag set')

        buy_info_from_file_inp = input("Load 'buy_info' from file ([y]/n)? ")
        if 'n' in buy_info_from_file_inp:
            buy_info_from_file = False

        wait_for_signal_to_change_inp = input("Wait for buy/sell signal to change (y/[n])? ")
        if 'y' in wait_for_signal_to_change_inp:
            wait_for_signal_to_change = True

    if not buy_info_from_file:
        print("Won't load 'buy_info' from file")
    else:
        buy_info_from_file = os.path.exists(buy_info_file)

        if not buy_info_from_file:
            print(f"{buy_info_file} doesn't exist")

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

    # DataFrame of relevant information
    #   - buy_state: dtypes = [None, bool]
    #   - buy_price: dtypes = [None, float]
    #   - trigger_name: dtypes = [None, str]
    #   - trigger_param: dtypes = [None, float]
    #   - weight: dtypes = [None, float]
    #   - trigger_order_id: dtypes = [None, int]
    buy_info = pd.DataFrame(
        columns=['buy_state', 'buy_price', 'trigger_name', 'trigger_param', 'weight', 'trigger_order_id'],
        index=strategy_keys
    )

    if buy_info_from_file:
        prev_buy_info = pd.read_csv(buy_info_file, index_col = 0)

        for i in buy_info.index:
            if i in prev_buy_info.index:
                buy_info.loc[i] = prev_buy_info.loc[i]


    for key in strategy_keys:
        buy_info.loc[key, 'weight'] = weights[key]

    if wait_for_signal_to_change:
        transaction_count = {}
        for key in strategy_keys:
            # TODO: process case m, m_bear = 1, 0 better
            transaction_count[key] = 1 if not pd.isna(buy_info.loc[key, 'buy_state']) else 0
    else:
        transaction_count = {k: 1 for k in strategy_keys}

    if debug:
        print(buy_info)
        print(transaction_count)
        print()
        # assert(False)

    prev_buy_state = {key: buy_info.loc[key, 'buy_state'] for key in strategy_keys}

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
                        if buy != prev_buy_state[key]:
                            prev_buy_state[key] = True
                            transaction_count[key] += 1

                            if transaction_count[key] > 1:
                                buy_info.loc[key, 'buy_state'] = True
                                buy_info.loc[key, 'buy_price'] = ftx_price(client, get_symbol(coin, m, bear = False))
                                buy_info.loc[key, 'trigger_name'] = sltp_args[0]
                                buy_info.loc[key, 'trigger_param'] = sltp_args[1]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan
                                changed = True

                    elif sell and len(sltp_args) == 4:
                        if (not sell) != prev_buy_state[key]:
                            prev_buy_state[key] = False
                            transaction_count[key] += 1

                            if transaction_count[key] > 1:
                                buy_info.loc[key, 'buy_state'] = False
                                buy_info.loc[key, 'buy_price'] = ftx_price(client, get_symbol(coin, m_bear, bear = True))
                                buy_info.loc[key, 'trigger_name'] = sltp_args[2]
                                buy_info.loc[key, 'trigger_param'] = sltp_args[3]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan
                                changed = True

                    elif sell and len(sltp_args) == 2:
                        if (not sell) != prev_buy_state[key]:
                            prev_buy_state[key] = False
                            transaction_count[key] += 1

                            if transaction_count[key] > 1:
                                buy_info.loc[key, 'buy_state'] = np.nan
                                buy_info.loc[key, 'buy_price'] = np.nan
                                buy_info.loc[key, 'trigger_name'] = np.nan
                                buy_info.loc[key, 'trigger_param'] = np.nan
                                buy_info.loc[key, 'trigger_order_id'] = np.nan
                                changed = True

                if debug:
                    print(buy_info)
                    print()

                # if changed:
                # TODO: if this works well, remove variable 'changed'
                buy_info = balance_portfolio(client, buy_info, debug = debug, verbose = debug)

            counter = (counter + gcd) % lcm

            wait(60 * 60 * gcd, timeTo0, verbose = debug)

            if debug:
                break
    except KeyboardInterrupt:
        error_flag = False
    finally:
        print('Exiting trading pipeline...')
        print()

        if error_flag and not debug:
            send_error_email(email_address, email_password)

        if not debug:
            buy_info.to_csv(buy_info_file)

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
