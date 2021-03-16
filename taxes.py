import numpy as np
import pandas as pd
import json
import glob
import time
import datetime
from keys import ftx_api_key, ftx_secret_key
from ftx.rest.client import FtxClient
from utils import *


def calculate_taxes(x):
    if x <= 30000:
        taxes = x * 0.3
    else:
        taxes = 30000 * 0.3 + (x - 30000) * 0.34

    return taxes


def merge_normal_and_conditional_trades_histories(
    client, trades_filename, conditional_trades_filename
):
    def _helper(filename, ignore_idx=None):
        trades = pd.read_csv("trading_logs/" + filename, index_col=0)
        trades["market"] = trades["market"].map(lambda x: x.replace("USDT", "USD"))

        if "triggeredAt" in trades.columns:
            trades["time"] = trades["triggeredAt"]
        else:
            trades["time"] = trades["createdAt"]

        if ignore_idx is not None:
            li = [order_id not in ignore_idx for order_id in trades["orderId"]]
            if np.sum(li) == 0:
                return 0, None
            trades = trades[li]

        return trades, trades["id"].values

    trades, ignore_idx = _helper(trades_filename)
    trades = [trades]

    conditional_trades, dummy = _helper(conditional_trades_filename, ignore_idx)

    if dummy is not None:
        # print(conditional_trades.shape)
        trades.append(conditional_trades)

    trades = pd.concat(trades, ignore_index=True).dropna(axis=1)

    # take current balances into account
    trades = trades.sort_values("time", ascending=False)

    _, balances = get_total_balance(client)

    for coin in balances.index:
        if "USD" not in coin:
            market = coin + "/USD"
            balance = balances.loc[coin, "total"]
            i = 0

            while balance > 0 and i < len(trades):
                idx = trades.index[i]

                while (
                    trades.loc[idx, "market"] != market
                    and i < len(trades)
                    and trades.loc[idx, "side"] != "buy"
                ):
                    i += 1
                    idx = trades.index[i]

                diff = min(balance, trades.loc[idx, "filledSize"])
                balance -= diff
                trades.loc[idx, "filledSize"] -= diff
                i += 1

    # TODO: change buys/sells such that diff = 0 for all markets?
    # for market in trades['market'].unique():
    #     if total_fill_sizes.index.isin([(market, 'sell')]).any():
    #         diff = total_fill_sizes[market, 'buy'] - total_fill_sizes[market, 'sell']

    return trades


# TODO: handle deposits/withdrawals
def get_taxes(
    trades_filename, conditional_trades_filename, year_end=None, verbose=True
):
    client = FtxClient(ftx_api_key, ftx_secret_key)

    trades = merge_normal_and_conditional_trades_histories(
        client, trades_filename, conditional_trades_filename
    )
    years = trades["time"].map(lambda x: int(x[:4]))

    year_start = years.min()
    max_year = years.max()

    if year_end is None:
        year_end = max_year
    year_end += 1
    if verbose:
        print(year_start, year_end)

    def _helper(trades, year, included_buys=None):
        trades["year"] = trades["time"].map(lambda x: int(x[:4]))
        trades = trades[trades["year"] == year]

        if included_buys is not None:
            trades = pd.concat([trades, included_buys], ignore_index=True)

        trades = trades.sort_values("time", ascending=False)

        total_fill_sizes = trades.groupby(["market", "side"])["filledSize"].sum()
        if verbose:
            print(total_fill_sizes)

        excluded_buys = []

        if year < max_year:
            for market in trades["market"].unique():
                if total_fill_sizes.index.isin([(market, "sell")]).any():
                    diff = (
                        total_fill_sizes[market, "buy"]
                        - total_fill_sizes[market, "sell"]
                    )
                else:
                    diff = total_fill_sizes[market, "buy"]
                assert diff >= 0

                if verbose:
                    print(market, diff, diff / total_fill_sizes[market, "buy"])

                li = (trades["market"] == market) & (trades["side"] == "buy")
                sub_idx = trades.index[li]

                for i in sub_idx:
                    excluded_buy = trades.loc[i].copy()

                    filledSize = excluded_buy["filledSize"]
                    diff_reduction = min(diff, filledSize)
                    diff -= diff_reduction

                    excluded_buy["filledSize"] = diff_reduction
                    trades.loc[i, "filledSize"] = filledSize - diff_reduction

                    excluded_buys.append(excluded_buy)

                    if diff == 0:
                        break

            excluded_buys = pd.concat(excluded_buys, axis=1).transpose()
            if verbose:
                print(
                    (excluded_buys["avgFillPrice"] * excluded_buys["filledSize"]).sum()
                )

        trades = trades[trades["filledSize"] > 0]
        trades = trades.sort_values("time", ascending=True)

        profits = {}

        # TODO: calculate value in euros using https://exchangeratesapi.io/
        for market in trades["market"].unique():
            effective_trades = []

            sub_trades = trades[trades["market"] == market]
            buy_trades = sub_trades[sub_trades["side"] == "buy"]
            sell_trades = sub_trades[sub_trades["side"] == "sell"].copy()

            for _, buy_trade in buy_trades.iterrows():
                size = buy_trade["filledSize"]

                trade = {"buy_value": buy_trade["avgFillPrice"] * size}

                sell_values = []
                potential_sell_trades = sell_trades[
                    sell_trades["time"] > buy_trade["time"]
                ]
                i = 0

                while size > 0 and i < len(potential_sell_trades):
                    idx = potential_sell_trades.index[i]

                    diff = min(size, sell_trades.loc[idx, "filledSize"])

                    size -= diff
                    sell_trades.loc[idx, "filledSize"] -= diff
                    sell_values.append(diff * sell_trades.loc[idx, "avgFillPrice"])

                    i += 1

                trade["sell_value"] = sum(sell_values)

                effective_trades.append(trade)

            profit = 0
            loss = 0

            for trade in effective_trades:
                trade_profit = trade["sell_value"] - trade["buy_value"]

                if trade_profit > 0:
                    profit += trade_profit
                else:
                    loss += trade_profit

            true_profit = profit + loss
            taxable_profit = profit

            if "BULL" not in market and "BEAR" not in market:
                # losses are eligible for exemption
                taxable_profit += loss

            taxes = calculate_taxes(taxable_profit)

            profits[market] = {
                "profit": profit,
                "loss": loss,
                "true_profit": true_profit,
                "taxable_profit": taxable_profit,
                "taxes": taxes,
            }

        aggregate_profit = {
            "profit": 0,
            "loss": 0,
            "true_profit": 0,
            "taxable_profit": 0,
            "taxes": 0,
        }

        for profit_values in profits.values():
            for k in profit_values.keys():
                aggregate_profit[k] += profit_values[k]

        return aggregate_profit, excluded_buys

    excluded_buys = None

    # TODO: turn this into a DataFrame; calculate precentage gains
    profits = {}

    for year in range(year_start, year_end):
        profit, excluded_buys = _helper(trades, year, included_buys=excluded_buys)
        if verbose:
            print(f"Taxable profit in year {year}: {profit}")
            print()

        profits[year] = profit

    return profits


if __name__ == "__main__":
    trades_filename = "trades.csv"
    conditional_trades_filename = "conditional_trades.csv"
    profits = get_taxes(trades_filename, conditional_trades_filename, verbose=False)
    print_dict(profits)
