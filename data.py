import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import requests
import glob

def get_and_save(coin, time_str):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&toTs=' + time_str + '&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'
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

def get_and_save_all():
    coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'BSV']

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
            url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'
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


def load_data(filename, batch_size, input_length, latency):
    obj = {}

    with open(filename, 'r') as file:
        obj = json.load(file)

    try:
        start_index = np.random.choice(len(obj['Data']) - batch_size - input_length - latency)
    except ValueError:
        start_index = 0

    data = obj['Data'][start_index:start_index + batch_size + input_length + latency]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key, value in item.items():
            if key != 'time':
                tmp.append(value)
        X[i, :] = tmp

    return X

if __name__ == "__main__":
    get_and_save_all()
