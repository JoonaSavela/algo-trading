import numpy as np
import requests
import json

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

    return opens, highs, lows, closes, volumes

def normalize_values(opens, highs, lows, closes, volumes):
    i = 0
    while opens[i] == 0 and i < opens.size:
        i += 1
    divider = opens[i] if i < opens.size else 1
    return opens / divider, highs / divider, lows / divider, closes / divider, volumes / 100000

def is_valid_interval(data, left, right):
    return not np.any(data[left:right] == 0)

def calculate_y(closes):
    close_min = closes.min()
    close_max = closes.max()
    current = closes[0]
    peak = 2 * (current - close_min) / (close_max - close_min) - 1
    percent_gain = close_max / current - 1
    percent_loss = close_min / current - 1
    return np.array([peak, percent_gain, percent_loss])

def transform_data(opens, highs, lows, closes, volumes, input_length):
    X = np.empty(shape=[0, input_length * 5]) # 5 = number of types of input variables
    Y = np.empty(shape=[0, 3])

    i = input_length - 1

    while i <= len(opens) - input_length:
        b = i + 1
        a = b - input_length
        c = i
        d = c + input_length

        if is_valid_interval(opens, a, d) and is_valid_interval(highs, a, d) and \
            is_valid_interval(lows, a, d) and is_valid_interval(closes, a, d) and \
            is_valid_interval(volumes, a, d):
            x = np.concatenate([opens[a:b], highs[a:b], lows[a:b], closes[a:b], volumes[a:b]])
            X = np.append(X, [x], axis = 0)
            Y = np.append(Y, np.reshape(calculate_y(closes[c:d]), [1, 3]), axis = 0)

        i += 1

    return X, Y
