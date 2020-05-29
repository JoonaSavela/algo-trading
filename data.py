import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import requests
import glob
from keys import cryptocompare_key
from utils import get_time

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

def get_and_save(coin, time_str):
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

coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC']

def get_and_save_all():
    for coin in coins:
        if not os.path.exists('data/' + coin):
            os.mkdir('data/' + coin)
        time_max = -1
        for filename in glob.glob('data/' + coin + '/*.json'):
            split1 = filename.split('/')
            split2 = split1[2].split('.')
            if int(split2[0]) > time_max:
                time_max = int(split2[0])

        if time_max != -1:
            time = time_max + 2000 * 60
        else:
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&api_key=' + cryptocompare_key
            request = requests.get(url)
            content = json.loads(request._content)
            time = content['TimeTo'] - 7 * 24 * 60 * 60 + 2000 * 60
            print(coin + ': No previous files found')

        count = 0

        while get_and_save(coin, str(time)):
            time += 2000 * 60
            count += 1

        print('Coin', coin, 'processed,', count, 'interval(s) added')
        print()


def load_data(filename, sequence_length, latency, window_size, k = 1, return_start_index = False):
    obj = {}

    with open(filename, 'r') as file:
        obj = json.load(file)

    try:
        # start_index = np.random.choice(len(obj['Data']) - sequence_length - latency - window_size + 1 - k + 1)
        start_index = 0
    except ValueError:
        start_index = 0

    data = obj['Data'][start_index:start_index + sequence_length + latency + window_size - 1 + k - 1]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key, value in item.items():
            if key != 'time':
                tmp.append(value)
        X[i, :] = tmp

    if return_start_index:
        return X, start_index

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
            X = load_data(filename, 2001, 0, 1)
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



if __name__ == "__main__":
    get_and_save_all()
