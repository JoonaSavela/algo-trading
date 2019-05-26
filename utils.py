import numpy as np
import requests
import json
from datetime import datetime, timedelta

def get_data(url):
    # get the data in json format
    request = requests.get(url)
    content = json.loads(request._content)
    time_series = content["Time Series (1min)"]

    # initialize outputs
    times = np.array([])
    opens = np.array([])
    highs = np.array([])
    lows = np.array([])
    closes = np.array([])
    volumes = np.array([])

    # reverse output ordering (starts from oldest, ends in newest)
    tmp = list(time_series.items())
    tmp.reverse()

    # insert the data into the outputs
    for key, item in dict(tmp).items():
        datetime_object = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        times = np.append(times, datetime_object)
        opens = np.append(opens, float(item["1. open"]))
        highs = np.append(highs, float(item["2. high"]))
        lows = np.append(lows, float(item["3. low"]))
        closes = np.append(closes, float(item["4. close"]))
        volumes = np.append(volumes, float(item["5. volume"]))

    return times, opens, highs, lows, closes, volumes

def normalize_prices(opens, highs, lows, closes):
    divider = opens[0]
    return opens / divider, highs / divider, lows / divider, closes / divider

def is_same_day(datetime1, datetime2):
    return datetime1.date() == datetime2.date()

def times_min_max(times):
    times_without_dates = list(map(lambda x: x.time(), list(times)))
    return min(times_without_dates), max(times_without_dates)

def is_valid_interval(dt, left, right, time_min, time_max, minutes = 1):
    datetime1 = dt - timedelta(minutes = left * minutes)
    datetime2 = dt + timedelta(minutes = right * minutes)
    return is_same_day(dt, datetime1) and is_same_day(dt, datetime2) and \
        datetime1.time() >= time_min and datetime2.time() <= time_max
