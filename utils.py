import numpy as np
import requests
import json
from datetime import datetime

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
