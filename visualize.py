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
from utils import *

# TODO: check if fetching of logs can be automated

def combine_files(main_filename):
    files = glob.glob('trading_logs/*.csv')

    dataFrames = [pd.read_csv(file) for file in files]

    # print([len(df) for df in dataFrames])

    df = pd.concat(dataFrames, sort = False, ignore_index = True).drop_duplicates('id')

    df = df.sort_values('time')

    print(len(df))

    df.to_csv('trading_logs/' + main_filename, index = False)

# TODO: handle new year
# Assumes no FTT is ever bought
# Assumes no new money is ever deposited
def get_taxable_profit(main_filename, year):
    trades = pd.read_csv('trading_logs/' + main_filename)
    trades['year'] = trades['time'].map(lambda x: int(x[:4]))
    trades = trades[trades['year'] == year]
    trades = trades.sort_values('time')

    # ignores first sells
    while (trades.head(1)['side'] == 'sell').all():
        trades.drop(trades.head(1).index, inplace=True)

    # ignores last buys
    while (trades.tail(1)['side'] == 'buy').all():
        trades.drop(trades.tail(1).index, inplace=True)

    li = trades['feeCurrency'] != 'USDT'
    trades.loc[li, 'fee'] = trades.loc[li, 'fee'] * trades.loc[li, 'price']
    trades['feeCurrency'] = 'USDT'

    profits_df = trades.groupby(['market', 'side']).apply(lambda x: (x['price'] * x['size']).sum())
    # print(profits_df)
    profits_df = profits_df.rename('total').reset_index()

    buys = profits_df[profits_df['side'] == 'buy']['total'].values
    sells = profits_df[profits_df['side'] == 'sell']['total'].values
    profits = sells - buys

    profit = profits.sum()
    total_fees = trades['fee'].sum()

    print(f'Taxable profit: {profit}')
    print(f'Fees payed: {total_fees}')


def plot_trades(main_filename, years = None, months = None, normalize = False):
    plt.style.use('seaborn')

    trades = pd.read_csv('trading_logs/' + main_filename)
    trades = trades[['time', 'market', 'side', 'size', 'price']]
    trades = trades.sort_values('time')

    # ignores first sells
    while (trades.head(1)['side'] == 'sell').all():
        trades.drop(trades.head(1).index, inplace=True)

    # ignores last buys
    while (trades.tail(1)['side'] == 'buy').all():
        trades.drop(trades.tail(1).index, inplace=True)

    if years is None and months is None:
        print(trades['time'].min()[:10])

    # TODO: group by chosen years and months

    trades['float_time'] = trades['time'].map(lambda x: time.mktime(datetime.datetime.strptime(x[:13], "%Y-%m-%dT%H").timetuple()))

    # TODO: DRY DRY
    usd_value = trades.groupby(['float_time', 'market', 'side']).apply(lambda x: (x['price'] * x['size']).sum())
    full_time = trades.groupby(['float_time', 'market', 'side'])['time'].min()

    # print(usd_value.groupby(['market', 'side']).sum())

    df = usd_value.rename('usdValue').reset_index()
    df = df.merge(full_time.reset_index())
    df = df.sort_values('time')

    if normalize:
        df['usdValue'] /= df['usdValue'].iloc[0]
    df['t'] = (df['float_time'] - df['float_time'].min()) / (60 * 60 * 24)
    xlab = 'days'
    n_months = df['t'].max() / 30

    if n_months > 2 * 12:
        df['t'] /= 365
        xlab = 'years'
    elif n_months > 3:
        df['t'] /= 30
        xlab = 'months'

    wealth = (df['usdValue'].iloc[-1] / df['usdValue'].iloc[0]) ** (1 / n_months)
    print(wealth, wealth ** 12)

    # print(df['usdValue'].iloc[-1])
    # print(df['usdValue'].iloc[0])
    # print(df['usdValue'].iloc[-1] - df['usdValue'].iloc[0])

    # test_files = glob.glob('data/ETH/*.json')
    # test_files.sort(key = get_time)

    # TODO: plot the price action between the events
    plt.plot(df['t'], df['usdValue'])
    plt.xlabel(xlab)
    plt.ylabel('Strategy')
    plt.show()


if __name__ == "__main__":
    main_filename = 'trades_all.csv'
    combine_files(main_filename)
    get_taxable_profit(main_filename, 2020)
    plot_trades(main_filename)
