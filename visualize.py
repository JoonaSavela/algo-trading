try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
import numpy as np
import pandas as pd
import json
import glob
import time
import datetime
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient

# TODO: implement
def aggregate_trades(year):
    pass

# TODO: group by chosen years and months
def plot_trades(years = None, months = None):
    trades = pd.read_csv('trading_logs/trades.csv')
    trades = trades[['time', 'market', 'side', 'size', 'price']]



    trades['float_time'] = trades['time'].map(lambda x: time.mktime(datetime.datetime.strptime(x[:13], "%Y-%m-%dT%H").timetuple()))

    usd_value = trades.groupby(['float_time', 'market', 'side']).apply(lambda x: (x['price'] * x['size']).sum())
    full_time = trades.groupby(['float_time', 'market', 'side'])['time'].min()

    df = usd_value.rename('usdValue').reset_index()
    df = df.merge(full_time.reset_index())
    df = df.sort_values('time')
    df = df[df['usdValue'] > 10]

    if months is None or 4 in months:
        df.loc[7, 'usdValue'] += df.loc[6, 'usdValue']
        df = df.drop(6)

    df['usdValue'] /= df['usdValue'].iloc[0]
    df['t'] = (df['float_time'] - df['float_time'].min()) / (60 * 60 * 24)
    xlab = 'days'
    n_months = df['t'].max() / 30
    # print(n_months)

    if n_months > 3:
        df['t'] /= 30
        xlab = 'months'

    wealth = df['usdValue'].iloc[-1] ** (1 / n_months)
    print(wealth, wealth ** 12)

    # TODO: plot the price action between the events
    plt.plot(df['t'], df['usdValue'])
    plt.xlabel(xlab)
    plt.ylabel('profit')
    plt.show()


if __name__ == "__main__":
    plot_trades()
