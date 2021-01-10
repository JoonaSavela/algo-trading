import json
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import os
import requests
import glob
from keys import cryptocompare_key
from utils import get_time, get_total_balance, get_symbol
import time
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient

def _get_recent_data(coin, TimeTo, size, type, aggregate):
    if type == 'm':
        url = 'https://min-api.cryptocompare.com/data/histominute'
    elif type == 'h':
        url = 'https://min-api.cryptocompare.com/data/histohour'
    else:
        raise ValueError('type must be "m" or "h"')
    url += '?fsym=' \
        + coin \
        + '&tsym=USD&limit=' \
        + str(size - 1) \
        + ('&toTs={}'.format(TimeTo) if TimeTo is not None else '') \
        + '&aggregate=' \
        + str(aggregate) \
        + '&api_key=' \
        + cryptocompare_key

    request = requests.get(url)
    content = request.json()

    data = content['Data']
    data_keys = ['close', 'high', 'low', 'open', 'volumefrom', 'volumeto']

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key in data_keys:
            tmp.append(item[key])
        X[i, :] = tmp

    if content['Response'] == 'Error':
        print(content)

    return X, content['TimeTo'], content['TimeFrom']


def get_recent_data(coin, size = 3 * 14, type = 'm', aggregate = 1):
    res = np.zeros(shape=(size, 6))

    s = size
    timeFrom = None

    while s > 0:
        if s == size:
            X, timeTo, timeFrom = _get_recent_data(coin, timeFrom, min(2000, s), type, aggregate)
        else:
            X, _, timeFrom = _get_recent_data(coin, timeFrom, min(2000, s), type, aggregate)

        s -= X.shape[0]

        res[s:s+X.shape[0], :] = X

    return res, timeTo

def get_and_save(coin, t):
    if t > time.time():
        return False
    time_str = str(t)
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&toTs=' + time_str + '&api_key=' + cryptocompare_key
    request = requests.get(url)
    content = json.loads(request._content)

    is_same_time = content['TimeTo'] == int(time_str)

    if content['Response'] == 'Success' and len(content['Data']) > 2000 and is_same_time:
        with open('data/' + coin + '/' + time_str + '.json', 'w') as file:
            json.dump(content, file)
    elif not is_same_time:
        # print('The "To" time was different than expected')
        pass
    elif len(content['Data']) <= 2000:
        print('Data length is under 2001')
    else:
        print('An error occurred')
        for key, item in content.items():
            print(key, item)
        print()

    return is_same_time

# TODO: move this somewhere more sensible
coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC']

def get_and_save_all():
    for coin in coins:
        if not os.path.exists('data/' + coin):
            os.mkdir('data/' + coin)
        t_max = -1
        for filename in glob.glob('data/' + coin + '/*.json'):
            split1 = filename.split('/')
            split2 = split1[2].split('.')
            if int(split2[0]) > t_max:
                t_max = int(split2[0])

        if t_max != -1:
            t = t_max + 2000 * 60
        else:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&api_key=' + cryptocompare_key
            request = requests.get(url)
            content = json.loads(request._content)
            t = content['TimeTo'] - 7 * 24 * 60 * 60 + 2000 * 60
            print(coin + ': No previous files found')

        count = 0

        while get_and_save(coin, t):
            t += 2000 * 60
            count += 1

        print('Coin', coin, 'processed,', count, 'interval(s) added')
    print()

# TODO: use pandas
def load_data(filename, sequence_length):
    obj = {}

    with open(filename, 'r') as file:
        obj = json.load(file)

    data = obj['Data'][:sequence_length]

    data_keys = ['close', 'high', 'low', 'open', 'volumefrom', 'volumeto']

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key in data_keys:
            tmp.append(item[key])
        X[i, :] = tmp

    return X

def load_all_data(filenames, index = 0, return_time = False):
    filenames = sorted(filenames, key = get_time)

    idx = np.arange(1, len(filenames))
    li = np.diff(np.array(list(map(lambda x: get_time(x), filenames)))) != 120000

    points = [0]
    for p in idx[li]:
        points.append(p)
    points.append(len(filenames))

    points = list(zip(points[:-1], points[1:]))

    if isinstance(index, int):
        # TODO: change this into an assertion
        index = min(index, len(points) - 1)
        idx = [index]
    elif not isinstance(index, list):
        raise ValueError('index must be either int or list')
    else:
        idx = index

    res = []

    for start, end in map(lambda x: points[x], idx):
        fnames = filenames[start:end]

        Xs = []

        for filename in fnames:
            X = load_data(filename, 2001)
            Xs.append(X[:2000, :]) # remove duplicates

        X = np.concatenate(Xs)
        res.append(X)

    if len(res) == 1:
        if return_time:
            return res[0], get_time(filenames[points[idx[0]][1] - 1])
        return res[0]

    if return_time:
        return res, list(map(lambda x: get_time(filenames[points[x][1] - 1]), idx))
    return res

def save_orderbook_data():
    source_symbol = 'USD'
    client = FtxClient(ftx_api_key, ftx_secret_key)
    for coin in coins:
        if not os.path.exists('data/orderbooks/' + coin):
            os.mkdir('data/orderbooks/' + coin)

        symbols = [
            coin,
            'BULL' if coin == 'BTC' else coin + 'BULL',
            'BEAR' if coin == 'BTC' else coin + 'BEAR',
            'HEDGE' if coin == 'BTC' else coin + 'HEDGE'
        ]

        for symbol in symbols:
            if not os.path.exists('data/orderbooks/' + coin + '/' + symbol):
                os.mkdir('data/orderbooks/' + coin + '/' + symbol)
            orderbook = client.get_orderbook(symbol + '/' + source_symbol, 100)
            time.sleep(0.05)

            filename = 'data/orderbooks/' + coin + '/' + symbol + '/' \
                + str(round(time.time())) + '.json'
            with open(filename, 'w') as fp:
                json.dump(orderbook, fp)

        print(coin, "orderbook data saved")
    print()


# TODO: is max (the output) the best?
# TODO: add comments
def get_average_spread(coin, m, total_balances, m_bear = None, step = 0.0001):
    bull_folder = get_symbol(coin, m, bear = False)
    foldernames_bull = [f'data/orderbooks/{coin}/{bull_folder}/']

    if (m_bear is not None) and (m_bear != 0):
        bear_folder = get_symbol(coin, m_bear, bear = True)

        foldernames_bear = [f'data/orderbooks/{coin}/{bear_folder}/']
    else:
        foldernames_bear = []

    foldernames = foldernames_bull + foldernames_bear

    average_spreads = [[] for total_balance in total_balances]

    for folder in foldernames:
        filenames = glob.glob(folder + '*.json')

        distribution_ask = []
        distribution_bid = []

        for filename in filenames:
            with open(filename, 'r') as fp:
                orderbook = json.load(fp)

            asks = np.array(orderbook['asks'])
            bids = np.array(orderbook['bids'])
            price = (asks[0, 0] + bids[0, 0]) / 2

            usd_value_asks = np.prod(asks, axis=1)
            usd_value_bids = np.prod(bids, axis=1)
            percentage_ask = asks[:, 0] / price - 1
            percentage_bid = bids[:, 0] / price - 1

            while len(distribution_ask) * step < np.max(percentage_ask) + step:
                distribution_ask.append(0.0)

            idx_ask = np.ceil(percentage_ask / step).astype(int)

            distribution_ask = np.array(distribution_ask)
            distribution_ask[idx_ask] += usd_value_asks / len(filenames)
            distribution_ask = list(distribution_ask)

            while -len(distribution_bid) * step > np.min(percentage_bid) - step:
                distribution_bid.append(0.0)

            idx_bid = np.ceil(np.abs(percentage_bid) / step).astype(int)

            distribution_bid = np.array(distribution_bid)
            distribution_bid[idx_bid] += usd_value_bids / len(filenames)
            distribution_bid = list(distribution_bid)

        distribution_ask = np.array(distribution_ask)
        percentage_ask = np.linspace(0, len(distribution_ask) * step, len(distribution_ask))

        distribution_bid = np.array(distribution_bid)
        percentage_bid = np.linspace(0, -len(distribution_bid) * step, len(distribution_bid))

        for i, total_balance in enumerate(total_balances):
            li_asks = np.cumsum(distribution_ask) < total_balance
            weights_ask = distribution_ask[li_asks] / total_balance
            if len(weights_ask) < len(percentage_ask):
                weights_ask = np.concatenate([weights_ask, [1 - np.sum(weights_ask)]])
            else:
                weights_ask /= np.sum(weights_ask)
            average_spread_ask = np.sum(percentage_ask[:len(weights_ask)] * weights_ask)
            average_spreads[i].append(np.abs(average_spread_ask))

            li_bids = np.cumsum(distribution_bid) < total_balance
            weights_bid = distribution_bid[li_bids] / total_balance
            if len(weights_bid) < len(percentage_bid):
                weights_bid = np.concatenate([weights_bid, [1 - np.sum(weights_bid)]])
            else:
                weights_bid /= np.sum(weights_bid)
            average_spread_bid = np.sum(percentage_bid[:len(weights_bid)] * weights_bid)
            average_spreads[i].append(np.abs(average_spread_bid))

    # return np.array(average_spreads)
    return np.max(average_spreads, axis=1)


def get_order_history(client, end_time):
    order_history = client.get_order_history(end_time = end_time)
    time.sleep(0.05)
    order_history = pd.DataFrame(order_history)
    return order_history

def get_conditional_order_history(client, end_time):
    conditional_order_history = client.get_conditional_order_history(end_time = end_time)
    time.sleep(0.05)
    conditional_order_history = pd.DataFrame(conditional_order_history)
    return conditional_order_history

def get_timestamps(time_string_series, time_string_format = "%Y-%m-%dT%H:%M:%S"):
    timestamps = time_string_series.map(
        lambda x: time.mktime(
            datetime.strptime(
                x[:len(time_string_format) + 2],
                time_string_format
            ).timetuple()
        )
    )
    return timestamps


def get_trade_history(client, get_order_history_f, max_time = None):
    if max_time is None:
        max_time = -1

    end_time = time.time()
    order_history = get_order_history_f(client, end_time)
    timestamps = get_timestamps(order_history['createdAt'])
    end_time = timestamps.min()

    res = [order_history]

    while len(order_history.index) == 100 and end_time > max_time:
        order_history = get_order_history_f(client, end_time)
        res.append(order_history)
        timestamps = get_timestamps(order_history['createdAt'])
        end_time = timestamps.min()

    res = pd.concat(res, ignore_index = True)
    res = res[res['filledSize'] > 0.]

    return res


def get_trades_data_frame_and_max_time(fname):
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col = 0)
        max_time = get_timestamps(df['createdAt']).max()
    else:
        df = None
        max_time = None

    return df, max_time



def save_trade_history():
    client = FtxClient(ftx_api_key, ftx_secret_key)
    trades_fname = 'trading_logs/trades.csv'
    conditional_trades_fname = 'trading_logs/conditional_trades.csv'

    def _helper(fname, get_order_history_f):
        prev_trades, max_time = get_trades_data_frame_and_max_time(fname)
        print(max_time)
        trades = [prev_trades] if prev_trades is not None else []

        trade_history = get_trade_history(client, get_order_history_f, max_time = max_time)

        trades.append(trade_history)
        trades = pd.concat(trades, ignore_index = True).drop_duplicates('id').sort_values('createdAt')

        trades.to_csv(fname)

    _helper(trades_fname, get_order_history)
    _helper(conditional_trades_fname, get_conditional_order_history)

    print()


def save_total_balance():
    client = FtxClient(ftx_api_key, ftx_secret_key)

    balance = pd.DataFrame([[
        time.time(),
        get_total_balance(client, separate = False),
    ]], columns=['time', 'balance'])
    print(float(balance['time']), float(balance['balance']))

    fname = 'trading_logs/balances.csv'
    balances = [balance]

    if os.path.exists(fname):
        balances.append(pd.read_csv(fname, index_col = 0))

    balances = pd.concat(balances, ignore_index = True).sort_values('time')
    balances.to_csv(fname)

    print()


# TODO: save deposits/withdrawals?
# TODO: better prints
if __name__ == "__main__":
    get_and_save_all()
    save_orderbook_data()
    save_trade_history()
    save_total_balance()
