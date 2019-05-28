import numpy as np
import requests
import json

def is_valid_interval(data, left = 0, right = None):
    right = len(data) if right is None else right
    return not np.any(data[left:right] == 0)

def fill_zeros(data):
    new_data = data
    n = len(data)
    for i in range(n):
        if data[i] == 0:
            if i == 0:
                k = i + 1
                while k < n and data[k] == 0:
                    k += 1
                if k < n:
                    new_data[i] = data[k]
            elif i == n - 1:
                j = i - 1
                while j >= 0 and data[j] == 0:
                    j -= 1
                if j >= 0:
                    new_data[i] = data[j]
            else:
                k = i + 1
                while k < n and data[k] == 0:
                    k += 1
                j = i - 1
                while j >= 0 and data[j] == 0:
                    j -= 1
                if k < n and j >= 0:
                    w1 = k - i
                    w2 = i - j
                    new_data[i] = (w1 * data[k] + w2 * data[j]) / (w1 + w2)

    return new_data

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
    if not is_valid_interval(opens):
        opens = fill_zeros(opens)

    if not is_valid_interval(highs):
        highs = fill_zeros(highs)

    if not is_valid_interval(lows):
        lows = fill_zeros(lows)

    if not is_valid_interval(closes):
        closes = fill_zeros(closes)

    if not is_valid_interval(volumes):
        volumes = fill_zeros(volumes)

    return opens, highs, lows, closes, volumes

def normalize_values(opens, highs, lows, closes, volumes):
    i = 0
    while opens[i] == 0 and i < opens.size:
        i += 1
    divider = opens[i] if i < opens.size else 1
    return opens / divider, highs / divider, lows / divider, closes / divider, volumes / 100000

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

        x = np.concatenate([opens[a:b], highs[a:b], lows[a:b], closes[a:b], volumes[a:b]])
        X = np.append(X, [x], axis = 0)
        Y = np.append(Y, np.reshape(calculate_y(closes[c:d]), [1, 3]), axis = 0)

        i += 1

    return X, Y
