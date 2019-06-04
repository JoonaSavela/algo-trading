import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import requests
import glob
from utils import is_valid_interval, fill_zeros, normalize_values, transform_data

def get_and_save_data(coin, time_str):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&limit=2000&toTs=' + time_str + '&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'
    request = requests.get(url)
    content = json.loads(request._content)
    print(len(content['Data']))
    for key, item in content.items():
        if key != 'Data':
            print(key, item)

    if content['Response'] == 'Success' and len(content['Data']) > 2000:
        with open('data/' + coin + '/' + time_str + '.json', 'w') as file:
            json.dump(content, file)
    elif len(content['Data']) < 2000:
        print('Data length is under 2000')

    print()

    return content['TimeTo']

def get_and_save_data_from_period():
    # coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC']
    coins = ['BTC', 'ETH']

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

        new_time = time - 2000 * 60
        old_time = new_time - 1

        while old_time < new_time:
            old_time = new_time
            new_time = get_and_save_data(coin, str(time))
            print('Period ' + str(time) + ' processed')
            time += 2000 * 60

        print('Coin', coin, 'processed')
        print()


def load_data():
    obj = {}

    for filename in glob.glob('data/*.json'):
        with open(filename, 'r') as file:
            obj[filename] = json.load(file)

    return obj

def load_and_transform_data(input_length = 4 * 14, test_split = 0):
    obj = load_data()

    X = np.empty(shape=[0, input_length * 5])
    Y = np.empty(shape=[0, 3])

    for day, day_data in obj.items():
        for stock_data in day_data.values():
            opens, highs, lows, closes, volumes = np.array(stock_data['opens']), \
                                                  np.array(stock_data['highs']), \
                                                  np.array(stock_data['lows']), \
                                                  np.array(stock_data['closes']), \
                                                  np.array(stock_data['volumes'])

            opens, highs, lows, closes, volumes = normalize_values(opens, highs, lows, closes, volumes)

            newX, newY = transform_data(opens, highs, lows, closes, volumes, input_length)

            X = np.append(X, newX, axis = 0)
            Y = np.append(Y, newY, axis = 0)

        print("Appended day " + day)
        print(X.shape)
        print(Y.shape)
        if (Y.shape[0] > 0):
            break

    return X, Y

if __name__ == "__main__":
    get_and_save_data()
