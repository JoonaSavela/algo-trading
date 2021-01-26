from keys import ftx_api_key, ftx_secret_key, email_address, email_password
from ftx.rest.client import FtxClient
import time
from datetime import datetime
from data import get_recent_data, get_conditional_order_history, get_timestamps
import numpy as np
import pandas as pd
from utils import aggregate as aggregate_F
from utils import floor_to_n, print_dict, get_gcd, get_lcm, round_to_n, \
    choose_get_buys_and_sells_fn, get_filtered_strategies_and_weights, \
    get_parameter_names, get_total_balance, apply_taxes, get_symbol
import json
import smtplib
from functools import reduce
import os
from collections import deque
from parameters import tax_rate

source_symbol = 'USD'
# TODO: if active_balancing == False, tie min_amount to minimum weight
min_amount = 0.01



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


# TODO: return the size obtained from the response of place_order
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

# TODO: return the size obtained from the response of place_order
def buy_assets(client, symbol, usd_size, round_n = 4, debug = False, return_price = False):
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

    if return_price:
        return size, price

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

def modify_trigger_order(
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
    raise NotImplementedError()
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

def get_weight_diffs(client, buy_info, symbols):
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

    # Get the target weights for the symbols
    for strategy_key in buy_info.index:
        weights[symbols[strategy_key]] += buy_info.loc[strategy_key, 'weight']

    target_weights = pd.Series(weights)

    # TODO: move this outside of the function
    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    missing_symbols = [k for k in target_weights.keys() if k not in balances.index]
    missing_balances = pd.DataFrame(index = missing_symbols, columns = balances.columns).fillna(0)
    balances = balances.append(missing_balances)

    actual_weights = balances['usdValue'] / total_balance
    weight_diffs = target_weights - actual_weights

    return weight_diffs, target_weights, actual_weights, total_balance, balances

def transfer_taxes_to_taxes_subaccount(client, size, debug = False):
    if not debug:
        client.transfer_to_subaccount(
            coin = source_symbol,
            size = size,
            source = None,
            destination = 'taxes'
        )
        time.sleep(0.05)

def append_to_dict_of_collections(d, k, v, collection_type = 'list'):
    if k not in d:
        if collection_type == 'list':
            d[k] = []
        elif collection_type == 'deque':
            d[k] = deque()
        else:
            raise ValueError('Invalid collection_type')

    d[k].append(v)

    return d

def get_buy_value_from_buy_history(buy_history, sell_size, verbose = False):
    buy_prices = []
    buy_sizes = []

    while sell_size > 0 and buy_history:
        buy_price, buy_size = buy_history.popleft()

        size_diff = min(sell_size, buy_size)
        sell_size -= size_diff
        buy_size -= size_diff

        buy_prices.append(buy_price)
        buy_sizes.append(size_diff)

        if buy_size > 0:
            buy_history.appendleft((buy_price, buy_size))

    if verbose:
        print(buy_history)
        print(buy_prices)
        print(buy_sizes)

    return np.dot(buy_prices, buy_sizes)

def open_buy_history(buy_history_fname):
    if os.path.exists(buy_history_fname):
        with open(buy_history_fname, 'r') as file:
            buy_history = json.load(file)

        for k, v in buy_history.items():
            buy_history[k] = deque(v)
    else:
        buy_history = {}

    return buy_history


def save_buy_history(buy_history, buy_history_fname):
    for k, v in buy_history.items():
        buy_history[k] = list(v)

    with open(buy_history_fname, 'w') as file:
        json.dump(buy_history, file)


def get_historical_price(client, symbol, last_signal_change_time):
    market = symbol + '/' + source_symbol

    X = client.get_historical_prices(
        market = market,
        resolution = 60,
        limit = 1,
        end_time = last_signal_change_time,
        # start_time = last_signal_change_time
    )
    time.sleep(0.05)

    return X[-1]['close']




# TODO: DRY DRY
def process_trades(
    client,
    buy_info,
    prev_buy_info,
    buy_history,
    triggered = None,
    place_take_profit_and_stop_loss_simultaneously = True,
    active_balancing = False,
    taxes = True,
    debug = False,
    verbose = False
):
    open_trigger_orders = get_conditional_orders(client)

    trigger_order_columns = [
        ('trigger_name', 'trigger_param', 'trigger_order_id')
    ]
    if place_take_profit_and_stop_loss_simultaneously:
        trigger_order_columns.append(
            ('trigger_name2', 'trigger_param2', 'trigger_order_id2')
        )

    if triggered is None:
        triggered = {k: False for k in buy_info.index}

    # get trigger prices for triggered orders
    prev_trigger_prices = {}
    prev_triggered_list = []

    for trigger_name_col, trigger_param_col, trigger_order_id_col in trigger_order_columns:
        prev_triggered = {}

        # detect triggered trigger orders
        for strategy_key in prev_buy_info.index:
            trigger_order_id = prev_buy_info.loc[strategy_key, trigger_order_id_col]

            if pd.isna(trigger_order_id):
                trigger_order_id = None
            else:
                trigger_order_id = str(int(trigger_order_id))

            if trigger_order_id is None:
                prev_triggered[strategy_key] = False
            else:
                prev_triggered[strategy_key] = \
                    trigger_order_id not in open_trigger_orders or \
                    open_trigger_orders[trigger_order_id]['status'] != 'open'

        prev_triggered_list.append(prev_triggered)

        for strategy_key in prev_buy_info.index:
            trigger_order_id = prev_buy_info.loc[strategy_key, trigger_order_id_col]
            if prev_triggered[strategy_key]:
                trigger_name = prev_buy_info.loc[strategy_key, trigger_name_col]
                if trigger_name == 'trailing':
                    end_time = time.time()
                    conditional_order_history = get_conditional_order_history(client, end_time)

                    while trigger_order_id not in conditional_order_history['id']:
                        if len(conditional_order_history.index) < 100:
                            raise ValueError("Trigger order not found")
                        timestamps = get_timestamps(conditional_order_history['createdAt'])
                        end_time = timestamps.min()
                        conditional_order_history = get_conditional_order_history(client, end_time)

                    idx = conditional_order_history.index
                    assert((conditional_order_history['id'] == trigger_order_i).sum() == 1)
                    i = idx[conditional_order_history['id'] == trigger_order_id][0]

                    prev_trigger_prices[strategy_key] = conditional_order_history.loc[
                        i, 'avgFillPrice'
                    ]

                else:
                    buy_price = prev_buy_info.loc[strategy_key, 'buy_price']
                    trigger_param = prev_buy_info.loc[strategy_key, trigger_param_col]
                    prev_trigger_prices[strategy_key] = trigger_param * buy_price

        # cancel trigger orders
        for strategy_key in prev_buy_info.index:
            trigger_order_id = prev_buy_info.loc[strategy_key, trigger_order_id_col]
            if not pd.isna(trigger_order_id) and not prev_triggered[strategy_key]:
                cancel_conditional_order(client, trigger_order_id, debug = debug)

    prev_triggered = {}

    for strategy_key in prev_buy_info.index:
        triggered_for_strategy = [t[strategy_key] for t in prev_triggered_list]
        assert(np.sum(triggered_for_strategy) <= 1)

        if len(triggered_for_strategy) == 1:
            prev_triggered[strategy_key] = triggered_for_strategy[0]
        else:
            prev_triggered[strategy_key] = reduce(
                lambda x, y: x or y,
                triggered_for_strategy
            )


    target_symbols = {}
    prev_symbols = {}
    for strategy_key in buy_info.index:
        changed = buy_info.loc[strategy_key, 'changed']

        if (prev_triggered[strategy_key] and not changed) or triggered[strategy_key]:
            symbol = source_symbol
        elif buy_info.loc[strategy_key, 'buy_state'] == True:
            symbol = get_symbol_from_key(strategy_key, bear = False)
        elif buy_info.loc[strategy_key, 'buy_state'] == False:
            symbol = get_symbol_from_key(strategy_key, bear = True)
        else:
            symbol = source_symbol

        target_symbols[strategy_key] = symbol

        if changed:
            if prev_triggered[strategy_key]:
                symbol = source_symbol
            elif prev_buy_info.loc[strategy_key, 'buy_state'] == True:
                symbol = get_symbol_from_key(strategy_key, bear = False)
            elif prev_buy_info.loc[strategy_key, 'buy_state'] == False:
                symbol = get_symbol_from_key(strategy_key, bear = True)
            else:
                symbol = source_symbol

        prev_symbols[strategy_key] = symbol

    if verbose:
        print(target_symbols)

    sell_prices = {}
    sell_sizes = {}

    # update usd value of each strategy
    for strategy_key in buy_info.index:
        if prev_triggered[strategy_key]:
            size = prev_buy_info.loc[strategy_key, 'size']
            buy_info.loc[strategy_key, 'usd_value'] = prev_trigger_prices[strategy_key] * size

            if buy_info.loc[strategy_key, 'changed']:
                sell_prices = append_to_dict_of_collections(sell_prices, prev_symbols[strategy_key], prev_trigger_prices[strategy_key])
                sell_sizes = append_to_dict_of_collections(sell_sizes, prev_symbols[strategy_key], size)

        elif prev_symbols[strategy_key] != source_symbol:
            buy_price = prev_buy_info.loc[strategy_key, 'buy_price']
            price = ftx_price(client, prev_symbols[strategy_key])

            size = prev_buy_info.loc[strategy_key, 'size']
            buy_info.loc[strategy_key, 'usd_value'] = price * size

    if not active_balancing:
        # update weights based on the current (updated) usd value
        buy_info['weight'] = buy_info['usd_value'] / buy_info['usd_value'].sum()

    weight_diffs, target_weights, actual_weights, total_balance, balances = get_weight_diffs(client, buy_info, target_symbols)

    if verbose:
        print(weight_diffs)

    # sell negatives
    li_neg = weight_diffs < 0

    for symbol, weight_diff in weight_diffs[li_neg].items():
        if symbol != source_symbol and np.abs(weight_diff) > min_amount:
            price = ftx_price(client, symbol)
            size = np.abs(weight_diff / actual_weights[symbol]) * balances['total'][symbol]
            size = sell_assets(client, symbol, size, round_n = 5, debug = debug)

            sell_prices = append_to_dict_of_collections(sell_prices, symbol, price)
            sell_sizes = append_to_dict_of_collections(sell_sizes, symbol, size)


    if taxes:
        assert(not active_balancing)

        total_amount_to_taxes = 0

        for symbol in sell_prices.keys():
            # sells need to be processed separately since they are separate transactions
            for sell_price, sell_size in zip(sell_prices[symbol], sell_sizes[symbol]):
                sell_value = sell_price * sell_size

                buy_value = get_buy_value_from_buy_history(
                    buy_history[symbol],
                    sell_size
                )

                if sell_value > buy_value:
                    amount_to_taxes = (sell_value - buy_value) * tax_rate

                    total_amount_to_taxes += amount_to_taxes

                    # update usd_value
                    strategy_subset_li = buy_info.loc[strategy_key, 'changed'] & \
                        (buy_info.index.map(lambda x: prev_symbols[x]) == symbol)
                    strategy_subset = buy_info.index[li]
                    total_weight_of_subset = buy_info.loc[strategy_subset, 'weight'].sum()

                    for strategy_key in strategy_subset:
                        conditional_weight = buy_info.loc[strategy_key, 'weight'] / total_weight_of_subset
                        buy_info.loc[strategy_key, 'usd_value'] -= conditional_weight * amount_to_taxes

        # transfer taxes into 'taxes' subaccount
        if total_amount_to_taxes > 0:
            transfer_taxes_to_taxes_subaccount(client, total_amount_to_taxes, debug = debug)

            # update weights
            buy_info['weight'] = buy_info['usd_value'] / buy_info['usd_value'].sum()

            # update weight diffs
            weight_diffs, target_weights, actual_weights, total_balance, balances = get_weight_diffs(client, buy_info, target_symbols)

    if not debug:
        time.sleep(1)

    # buy positives
    li_pos = ~li_neg

    for symbol, weight_diff in weight_diffs[li_pos].items():
        if symbol != source_symbol and weight_diff > min_amount:
            usd_size = weight_diff * total_balance
            size, price = buy_assets(client, symbol, usd_size, round_n = 5, debug = debug, return_price = True)

            buy_history = append_to_dict_of_collections(
                buy_history,
                symbol,
                (price, size),
                collection_type = 'deque'
            )

    if not debug:
        time.sleep(1)

    total_balance, balances = get_total_balance(client, separate = True, filter = list(target_weights.keys()))

    # (re)apply trigger orders if trigger_name is not 'trailing' or if trigger_order_id is None
    # else modify order s.t. its weight is correct
    for strategy_key, symbol in target_symbols.items():
        if symbol != source_symbol:
            size = buy_info.loc[strategy_key, 'weight'] / target_weights[symbol] * balances['total'][symbol]
            buy_info.loc[strategy_key, 'size'] = size

            for trigger_name_col, trigger_param_col, trigger_order_id_col in trigger_order_columns:
                # TODO: proces case when active_balancing = True correctly
                id, size = place_trigger_order(
                    client,
                    symbol,
                    buy_info.loc[strategy_key, 'buy_price'],
                    size,
                    buy_info.loc[strategy_key, trigger_name_col],
                    buy_info.loc[strategy_key, trigger_param_col],
                    round_n = 5,
                    debug = debug
                )

                id = str(int(id)) if id is not None else id
                buy_info.loc[strategy_key, trigger_order_id_col] = id

                if verbose:
                    print(id, type(id))
                    print(f'Placed order, size = {size}')

    # print all that was done
    print(buy_info.transpose())
    print_dict(buy_history)
    print()

    return buy_info, buy_history


# TODO: DRY DRY
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

            # TODO: make this more logical?
            if not triggered[strategy_key] and buy_info.loc[strategy_key, 'trigger_name'] == 'trailing':
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

                    id, size = modify_trigger_order(
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

    # cancel orders if their type is not 'trailing'
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

                id, size = modify_trigger_order(
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
    ask_for_input = True,
    place_take_profit_and_stop_loss_simultaneously = True,
    active_balancing = False, # TODO: move into optim_results folder, or remove altogether
    taxes = True
):
    print('Starting trading pipeline...')
    print()

    client = FtxClient(ftx_api_key, ftx_secret_key)

    debug = False
    buy_info_from_file = True
    buy_info_file = 'optim_results/buy_info.csv'

    if ask_for_input:
        debug_inp = input('Is this a debug run (y/[n])? ')
        if 'y' in debug_inp:
            debug = True

        if debug:
            print('Debug flag set')

        buy_info_from_file_inp = input("Load 'buy_info' from file ([y]/n)? ")
        if 'n' in buy_info_from_file_inp:
            buy_info_from_file = False

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

    print_dict(strategies)

    get_buys_and_sells_fns = {}
    N_parameters = {}

    for strategy_key in strategies.keys():
        strategy_type = strategy_key.split('_')[2]

        N_parameters[strategy_key] = len(get_parameter_names(strategy_type))
        get_buys_and_sells_fns[strategy_key] = choose_get_buys_and_sells_fn(strategy_type)


    if os.path.exists('optim_results/displacements_and_last_signal_change_time.json'):
        with open('optim_results/displacements_and_last_signal_change_time.json', 'r') as file:
            displacements_and_last_signal_change_time = json.load(file)

    displacements = {}
    last_signal_change_times = {}
    triggered = {}

    for k in strategies.keys():
        displacements[k] = int(displacements_and_last_signal_change_time[k][0])
        last_signal_change_times[k] = displacements_and_last_signal_change_time[k][2]
        triggered[k] = displacements_and_last_signal_change_time[k][3]

    displacement_values = np.sort(np.array([v for v in displacements.values()]))

    strategy_keys = np.array(sorted(strategies.keys(), key = lambda x: displacements[x]))

    aggregate_Ns_all = np.array([strategies[key]['params'][0] for key in strategy_keys])
    lcm = reduce(lambda x, y: get_lcm(x, y), aggregate_Ns_all)
    gcd = reduce(lambda x, y: get_gcd(x, y), aggregate_Ns_all)
    counter = 0

    # buy_info: DataFrame of relevant information
    if place_take_profit_and_stop_loss_simultaneously:
        buy_info = pd.DataFrame(
            columns=['buy_state', 'buy_price', 'trigger_name', 'trigger_param', 'weight', 'size',
                'usd_value', 'trigger_order_id', 'changed', 'trigger_name2', 'trigger_param2', 'trigger_order_id2'],
            index=strategy_keys
        )
    else:
        buy_info = pd.DataFrame(
            columns=['buy_state', 'buy_price', 'trigger_name', 'trigger_param', 'weight', 'size',
                'usd_value', 'trigger_order_id', 'changed'],
            index=strategy_keys
        )

    if buy_info_from_file:
        prev_buy_info = pd.read_csv(buy_info_file, index_col = 0)

        strategies_match_exactly = len(prev_buy_info.index) == len(buy_info.index)

        for i in prev_buy_info.index:
            if i not in buy_info.index:
                strategies_match_exactly = False
            for c in prev_buy_info.columns:
                if i in buy_info.index and c in buy_info.columns:
                    buy_info.loc[i, c] = prev_buy_info.loc[i, c]

    initial_action_flag = False

    if not buy_info_from_file or not strategies_match_exactly:
        initial_action_flag = True

        total_balance = get_total_balance(client, separate = False)

        for key in strategy_keys:
            buy_info.loc[key, 'weight'] = weights[key]
            buy_info.loc[key, 'usd_value'] = buy_info.loc[key, 'weight'] * total_balance

    buy_history_fname = 'buy_history.json'
    buy_history = open_buy_history(buy_history_fname)

    if debug:
        print(buy_info.transpose())
        print_dict(buy_history)
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

                buy_info['changed'] = False
                prev_buy_info = buy_info.copy()

                for key in keys:
                    coin = key.split('_')[0]
                    m, m_bear = tuple([int(x) for x in key.split('_')[3:]])

                    args = strategies[key]['params'][:N_parameters[key]]
                    sltp_args = strategies[key]['params'][N_parameters[key]:]
                    aggregate_N = args[0]

                    if place_take_profit_and_stop_loss_simultaneously:
                        short = 'short' in sltp_args
                    else:
                        short = len(sltp_args) == 4

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

                            buy_info.loc[key, 'buy_state'] = True

                            if initial_action_flag:
                                last_signal_change_time = last_signal_change_times[key] + \
                                    (time.time() - last_signal_change_times[key]) % (aggregate_N * 60 * 60)
                                buy_info.loc[key, 'buy_price'] = \
                                    get_historical_price(client, get_symbol(coin, m, bear = False), last_signal_change_time)
                            else:
                                buy_info.loc[key, 'buy_price'] = ftx_price(client, get_symbol(coin, m, bear = False))

                            buy_info.loc[key, 'size'] = np.nan
                            if place_take_profit_and_stop_loss_simultaneously:
                                buy_info.loc[key, 'trigger_name'] = 'take_profit'
                                buy_info.loc[key, 'trigger_param'] = sltp_args[1]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan

                                buy_info.loc[key, 'trigger_name2'] = sltp_args[2]
                                buy_info.loc[key, 'trigger_param2'] = sltp_args[3]
                                buy_info.loc[key, 'trigger_order_id2'] = np.nan
                            else:
                                buy_info.loc[key, 'trigger_name'] = sltp_args[0]
                                buy_info.loc[key, 'trigger_param'] = sltp_args[1]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan

                            buy_info.loc[key, 'changed'] = True

                    elif sell and short:
                        if (not sell) != prev_buy_state[key]:
                            prev_buy_state[key] = False

                            buy_info.loc[key, 'buy_state'] = False

                            if initial_action_flag:
                                last_signal_change_time = last_signal_change_times[key] + \
                                    (time.time() - last_signal_change_times[key]) % (aggregate_N * 60 * 60)
                                buy_info.loc[key, 'buy_price'] = \
                                    get_historical_price(client, get_symbol(coin, m_bear, bear = True), last_signal_change_time)
                            else:
                                buy_info.loc[key, 'buy_price'] = \
                                    ftx_price(client, get_symbol(coin, m_bear, bear = True))

                            buy_info.loc[key, 'size'] = np.nan
                            if place_take_profit_and_stop_loss_simultaneously:
                                buy_info.loc[key, 'trigger_name'] = 'take_profit'
                                buy_info.loc[key, 'trigger_param'] = sltp_args[5]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan

                                buy_info.loc[key, 'trigger_name2'] = sltp_args[6]
                                buy_info.loc[key, 'trigger_param2'] = sltp_args[7]
                                buy_info.loc[key, 'trigger_order_id2'] = np.nan
                            else:
                                buy_info.loc[key, 'trigger_name'] = sltp_args[2]
                                buy_info.loc[key, 'trigger_param'] = sltp_args[3]
                                buy_info.loc[key, 'trigger_order_id'] = np.nan

                            buy_info.loc[key, 'changed'] = True

                    elif sell and not short:
                        if (not sell) != prev_buy_state[key]:
                            prev_buy_state[key] = False

                            buy_info.loc[key, 'buy_state'] = np.nan
                            buy_info.loc[key, 'buy_price'] = np.nan
                            buy_info.loc[key, 'size'] = np.nan
                            buy_info.loc[key, 'trigger_name'] = np.nan
                            buy_info.loc[key, 'trigger_param'] = np.nan
                            buy_info.loc[key, 'trigger_order_id'] = np.nan

                            if place_take_profit_and_stop_loss_simultaneously:
                                buy_info.loc[key, 'trigger_name2'] = np.nan
                                buy_info.loc[key, 'trigger_param2'] = np.nan
                                buy_info.loc[key, 'trigger_order_id2'] = np.nan

                            buy_info.loc[key, 'changed'] = True

                if active_balancing or buy_info['changed'].any():
                    buy_info, buy_history = process_trades(
                        client,
                        buy_info,
                        prev_buy_info,
                        buy_history,
                        triggered,
                        place_take_profit_and_stop_loss_simultaneously,
                        active_balancing,
                        taxes = taxes,
                        debug = debug,
                        verbose = debug
                    )

            counter = (counter + gcd) % lcm

            wait(60 * 60 * gcd, timeTo0, verbose = debug)
            initial_action_flag = False
            triggered = None

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
            save_buy_history(buy_history, buy_history_fname)

        # uncomment this to ignore error and resume algorithm
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
