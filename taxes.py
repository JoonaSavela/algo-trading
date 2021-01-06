import numpy as np
import pandas as pd
import json
import glob
import time
import datetime
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from utils import *

def merge_normal_and_conditional_trades_histories(trades_filename, conditional_trades_filename):
    def _helper(filename, ignore_idx = None):
        trades = pd.read_csv('trading_logs/' + filename, index_col = 0)
        trades['market'] = trades['market'].map(
            lambda x: x.replace('USDT', 'USD')
        )

        if 'triggeredAt' in trades.columns:
            trades['time'] = trades['triggeredAt']
        else:
            trades['time'] = trades['createdAt']

        if ignore_idx is not None:
            li = [order_id not in ignore_idx for order_id in trades['orderId']]
            if np.sum(li) == 0:
                return 0, None
            trades = trades[li]

        return trades, trades['id'].values

    trades, ignore_idx = _helper(trades_filename)
    trades = [trades]

    conditional_trades, dummy = _helper(conditional_trades_filename, ignore_idx)

    if dummy is not None:
        # print(conditional_trades.shape)
        trades.append(conditional_trades)

    return pd.concat(trades, ignore_index = True).dropna(axis = 1)

# TODO: group events wisely
# TODO: handle new year
# TODO: handle deposits/withdrawals
def get_taxable_profit(trades_filename, conditional_trades_filename, year_start = None, year_end = None, verbose = True):
    trades = merge_normal_and_conditional_trades_histories(trades_filename, conditional_trades_filename)
    years = trades['time'].map(lambda x: int(x[:4]))

    max_year = years.max()

    if year_end is None:
        if year_start is None:
            year_start = years.min()
            year_end = max_year
        else:
            year_end = year_start
    year_end += 1
    if verbose:
        print(year_start, year_end)

    def _helper(trades, year, included_buys = None):
        trades['year'] = trades['time'].map(lambda x: int(x[:4]))
        trades = trades[trades['year'] == year]

        if included_buys is not None:
            # print(included_buys)
            trades = pd.concat(
                [
                    trades,
                    included_buys
                ],
                ignore_index = True
            )

        trades = trades.sort_values('time', ascending = False)

        total_fill_sizes = trades.groupby(['market', 'side'])['filledSize'].sum()
        if verbose:
            print(total_fill_sizes)

        excluded_buys = []

        if year < max_year:
            for market in trades['market'].unique():
                if total_fill_sizes.index.isin([(market, 'sell')]).any():
                    diff = total_fill_sizes[market, 'buy'] - total_fill_sizes[market, 'sell']
                else:
                    diff = total_fill_sizes[market, 'buy']
                assert(diff >= 0)

                if verbose:
                    print(market, diff)

                li = (trades['market'] == market) & (trades['side'] == 'buy')
                sub_idx = trades.index[li]

                for i in sub_idx:
                    excluded_buy = trades.loc[i].copy()

                    filledSize = excluded_buy['filledSize']
                    diff_reduction = min(diff, filledSize)
                    diff -= diff_reduction

                    excluded_buy['filledSize'] = diff_reduction
                    trades.loc[i, 'filledSize'] = filledSize - diff_reduction

                    excluded_buys.append(excluded_buy)

                    if diff == 0:
                        break

            excluded_buys = pd.concat(excluded_buys, axis = 1).transpose()
            if verbose:
                print((excluded_buys['avgFillPrice'] * excluded_buys['filledSize']).sum())

        trades = trades[trades['filledSize'] > 0]

        profits_series = trades.groupby(['market', 'side']).apply(lambda x: (x['avgFillPrice'] * x['filledSize']).sum())

        profits_series = profits_series.rename('total_USD_traded')
        if verbose:
            print(profits_series)

        buys = profits_series[:, 'buy'].sum()
        sells = profits_series[:, 'sell'].sum()

        profit = sells - buys

        return profit, excluded_buys

    excluded_buys = None

    profits = {}

    for year in range(year_start, year_end):
        profit, excluded_buys = _helper(trades, year, included_buys = excluded_buys)
        if verbose:
            print(f'Taxable profit in year {year}: {profit}')
            print()

        profits[year] = profit

    return profits


if __name__ == "__main__":
    trades_filename = 'trades.csv'
    conditional_trades_filename = 'conditional_trades.csv'
    profits = get_taxable_profit(trades_filename, conditional_trades_filename)
    print_dict(profits)
