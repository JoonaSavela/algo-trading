import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os
import requests
import glob
from utils import is_valid_interval, fill_zeros, normalize_values, transform_data


def get_data(url):
    # get the data in json format
    request = requests.get(url)
    content = json.loads(request._content)

    # initialize outputs
    opens = np.zeros(len(content))
    highs = np.zeros(len(content))
    lows = np.zeros(len(content))
    closes = np.zeros(len(content))
    volumes = np.zeros(len(content))

    # insert the data into the outputs
    for i in range(len(content)):
        item = content[i]
        try:
            opens[i] = float(item["open"])
            highs[i] = float(item["high"])
            lows[i] = float(item["low"])
            closes[i] = float(item["close"])
            volumes[i] = float(item["volume"])
        except TypeError:
            pass

    # fill in the zeros
    # if not is_valid_interval(opens):
    #     opens = fill_zeros(opens)
    #
    # if not is_valid_interval(highs):
    #     highs = fill_zeros(highs)
    #
    # if not is_valid_interval(lows):
    #     lows = fill_zeros(lows)
    #
    # if not is_valid_interval(closes):
    #     closes = fill_zeros(closes)
    #
    # if not is_valid_interval(volumes):
    #     volumes = fill_zeros(volumes)

    return opens, highs, lows, closes, volumes

def get_and_save_data(date_str = date.today().strftime('%Y%m%d')):
    # stocks = pd.read_csv('stocks.csv')
    coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC']

    obj = {}

    for coin in coins:
        url = 'https://min-api.cryptocompare.com/data/histominute?fsym=' + coin + '&tsym=USD&api_key=7038987dbc91dc65168c7d8868c69c742acc11e682ba67d6c669e236dbd85deb'
        try:
            opens, highs, lows, closes, volumes = get_data(url)

            stock_data = {
                'opens': opens.tolist(),
                'highs': highs.tolist(),
                'lows': lows.tolist(),
                'closes': closes.tolist(),
                'volumes': volumes.tolist()
            }

            obj[stock] = stock_data

            print("Appended " + stock)
        except KeyError:
            pass
        # break

    with open('data/' + date_str + '.json', 'w') as file:
        json.dump(obj, file)

def get_and_save_data_from_period(days = 30, replace = False):
    current_day = date.today() - timedelta(days=days)

    while current_day < date.today():
        date_str = current_day.strftime('%Y%m%d')
        filename = 'data/' + date_str + '.json'
        if not ( replace == False and os.path.isfile(filename) ):
            get_and_save_data(date_str)
            print('Day ' + date_str + ' processed.')
        else:
            print('File ' + filename + ' already exists.')


        current_day += timedelta(days=1)

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