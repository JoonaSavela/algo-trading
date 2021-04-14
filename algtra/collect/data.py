import os
import sys

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import requests
import glob
from ftx.rest.client import FtxClient
import keys
from algtra import utils
from algtra import constants
import time
from ciso8601 import parse_datetime
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from numba import njit


def get_filtered_markets(client, filter_volume=False):
    markets = pd.DataFrame(client.list_markets())
    time.sleep(0.05)

    markets = markets[markets["name"].str.contains("/USD")]
    markets = markets[markets["quoteCurrency"] == "USD"]
    markets = markets[~markets["name"].str.contains("USDT")]
    for curr in constants.NON_USD_FIAT_CURRENCIES:
        markets = markets[~markets["name"].str.contains(curr)]
    markets = markets[markets["tokenizedEquity"].isna()]
    if filter_volume:
        markets = markets[markets["volumeUsd24h"] > 0]

    markets["coin"] = markets["baseCurrency"].map(lambda x: utils.get_coin(x))
    markets = markets.sort_values("volumeUsd24h", ascending=False)

    return markets


# TODO: intead of limit=5000, calculate the optimal limit based on prev_end_time
# TODO: add option for hourly data
def get_price_data(client, market, limit=None, prev_end_time=None, verbose=False):
    res = []

    X = client.get_historical_prices(
        market=market,
        resolution=60,
        limit=5000 if limit is None else min(limit, 5000),
    )
    time.sleep(0.05)
    if limit is not None:
        limit -= len(X)
    res.extend(X)
    count = 1

    start_time = min(parse_datetime(x["startTime"]) for x in X)
    if verbose:
        print(count, len(X), start_time)
    start_time = start_time.timestamp()

    while (
        # count <= 10 and
        len(X) > 1
        and (limit is None or limit > 0)
        and (prev_end_time is None or prev_end_time < start_time)
    ):
        X = client.get_historical_prices(
            market=market,
            resolution=60,
            limit=5000 if limit is None else min(limit, 5000),
            end_time=start_time,
        )
        time.sleep(0.05)
        if limit is not None:
            limit -= len(X)
        res.extend(X)
        count += 1

        start_time = min(parse_datetime(x["startTime"]) for x in X)
        if verbose:
            print(count, len(X), start_time)
        start_time = start_time.timestamp()

    res = pd.DataFrame(res).drop_duplicates("startTime").sort_values("startTime")
    if prev_end_time is not None:
        res = res[
            res["startTime"].map(lambda x: datetime.timestamp(parse_datetime(x)))
            > prev_end_time
        ]

    return res


def load_price_data(
    data_dir, market, return_price_data=True, return_price_data_only=False
):
    symbol = market.split("/")[0]
    coin = utils.get_coin(symbol)

    data_files = glob.glob(os.path.join(data_dir, coin, symbol + "_*_*.csv"))
    assert len(data_files) <= 1, "Multiple possible data files found."

    price_data = None
    prev_end_time = None
    data_length = 0

    if len(data_files) == 1:
        data_file = data_files[0]
        if return_price_data:
            price_data = pd.read_csv(data_file, index_col=0).reset_index()
            data_length = len(price_data)

            price_data["time"] = price_data["startTime"].map(
                lambda x: datetime.timestamp(parse_datetime(x))
            )
            prev_end_time = price_data["time"].max()
        else:
            data_file_split = data_file.split("_")
            prev_end_time = int(data_file_split[1])
            data_length = int(data_file_split[-1].split(".csv")[0])

    if return_price_data:
        if return_price_data_only:
            return price_data

        return price_data, prev_end_time, data_length

    return prev_end_time, data_length


def split_price_data(price_data):
    price_data_splits = []

    time_diffs = np.diff(price_data["time"].values) // 60

    idx = np.argwhere(time_diffs > constants.PRICE_DATA_MAX_GAP).reshape((-1,)) + 1
    idx = np.concatenate([np.array([0]), idx, np.array([len(price_data)])])

    for start, end in zip(idx[:-1], idx[1:]):
        if end - start > constants.MIN_AGGREGATE_N * constants.MIN_W * 60:
            sub_idx = price_data.index[start:end]
            price_data_splits.append(price_data.loc[sub_idx])

    return price_data_splits


def load_data_for_coin(data_dir, coin, discard_half_leverage=True):
    coin_dir = os.path.abspath(os.path.join(data_dir, coin))
    coin_spread_dir = os.path.abspath(os.path.join(data_dir, "spreads", coin))

    price_data = {}
    spread_data = {}

    for filename in glob.glob(coin_dir + "/*.csv"):
        symbol = filename.replace(coin_dir + "/", "").split("_")[0]
        leverage = utils.get_leverage_from_symbol(symbol)

        if discard_half_leverage and leverage == 0.5:
            continue

        spread_file = os.path.abspath(os.path.join(coin_spread_dir, f"{symbol}.json"))
        assert os.path.exists(
            spread_file
        ), f"A corresponding spread file should exist for {symbol}."

        market = symbol + "/USD"
        symbol_price_data = load_price_data(
            data_dir, market, return_price_data_only=True
        )
        symbol_price_data_splits = split_price_data(symbol_price_data)
        price_data[leverage] = symbol_price_data_splits

        symbol_spread_data = load_spread_distributions(data_dir, symbol, stack=True)
        spread_data[leverage] = symbol_spread_data

    return price_data, spread_data


def save_price_data(data_dir):
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    markets = get_filtered_markets(client)

    datapoints_added = 0

    max_data_length = 0
    max_symbol = None

    print("Saving price data...")

    for coin, markets_group in tqdm(markets.groupby("coin")):
        coin_dir = os.path.join(data_dir, coin)
        if not os.path.exists(coin_dir):
            os.mkdir(coin_dir)

        for market in markets_group["name"]:
            prev_end_time, prev_data_length = load_price_data(
                data_dir, market, return_price_data=False
            )

            new_price_data = get_price_data(
                client,
                market,
                limit=None,
                prev_end_time=prev_end_time,
                verbose=False,
            )

            symbol = market.split("/")[0]

            price_data = new_price_data.drop_duplicates("startTime").sort_values(
                "startTime"
            )
            end_time = (
                price_data["startTime"]
                .map(lambda x: datetime.timestamp(parse_datetime(x)))
                .max()
            )

            datapoints_added += len(price_data)
            data_length = prev_data_length + len(price_data)

            if data_length > max_data_length:
                max_data_length = data_length
                max_symbol = symbol

            data_file_exists = prev_end_time is not None
            if data_file_exists:
                prev_data_file = os.path.join(
                    coin_dir, symbol + f"_{int(prev_end_time)}_{prev_data_length}.csv"
                )
                assert os.path.exists(prev_data_file)

            new_data_file = os.path.join(
                coin_dir, symbol + f"_{int(end_time)}_{data_length}.csv"
            )
            price_data.to_csv(
                prev_data_file if data_file_exists else new_data_file,
                mode="a" if data_file_exists else "w",
                header=not data_file_exists,
            )
            if data_file_exists:
                os.rename(prev_data_file, new_data_file)

    print(f"Saved {datapoints_added} price data points")
    print(f"Max. data length: {max_data_length} (symbol {max_symbol})")
    print()


def clean_orderbook(orderbook, debug=False):
    flag = debug

    while (
        orderbook["asks"]
        and orderbook["bids"]
        and (orderbook["asks"][0][0] < orderbook["bids"][0][0] or flag)
    ):
        size = min(orderbook["asks"][0][1], orderbook["bids"][0][1])

        for side in constants.ASKS_AND_BIDS:
            orderbook[side][0][1] -= size

            if orderbook[side][0][1] == 0.0:
                orderbook[side] = orderbook[side][1:]

        flag = False

    return orderbook


def orderbook_size_diff(orderbook):
    total_ask_size = sum(ask[1] for ask in orderbook["asks"])
    total_bid_size = sum(bid[1] for bid in orderbook["bids"])

    return total_ask_size - total_bid_size


def get_spread_distributions(client, market):
    orderbook = client.get_orderbook(market, 100)
    orderbook = clean_orderbook(orderbook)
    time.sleep(0.05)

    asks = np.array(orderbook["asks"])
    bids = np.array(orderbook["bids"])
    side_arrays = {"asks": asks, "bids": bids}
    price = (asks[0, 0] + bids[0, 0]) / 2

    distributions = {"N": 1}

    for side in constants.ASKS_AND_BIDS:
        usd_values = np.prod(orderbook[side], axis=1)
        log_percentages = np.log(side_arrays[side][:, 0] / price)

        try:
            assert np.all(np.diff(np.abs(log_percentages)) >= 0)
        except AssertionError:
            logger.error(market)
            logger.error(log_percentages)
            raise

        distribution = np.zeros(
            int(np.max(np.abs(log_percentages)) / constants.ORDERBOOK_STEP_SIZE + 2)
        )
        idx = np.ceil(np.abs(log_percentages) / constants.ORDERBOOK_STEP_SIZE).astype(
            int
        )

        for i in range(len(idx)):
            distribution[idx[i]] += usd_values[i]

        distributions[side] = distribution

    return distributions


def load_spread_distributions(data_dir, symbol, stack=False):
    coin = utils.get_coin(symbol)
    coin_dir = os.path.join(data_dir, "spreads", coin)

    data_file = os.path.join(coin_dir, symbol + ".json")

    # Default if file doesn't exist
    distributions = {"N": 0}
    for side in constants.ASKS_AND_BIDS:
        distributions[side] = np.array([0])

    if os.path.exists(data_file):
        with open(data_file, "r") as file:
            distributions = json.load(file)

        for side in constants.ASKS_AND_BIDS:
            distributions[side] = np.array(distributions[side])

    if stack:
        max_N = np.max([len(distributions[side]) for side in constants.ASKS_AND_BIDS])

        res = np.zeros((max_N, 2))

        for i in range(2):
            distr = distributions[constants.ASKS_AND_BIDS[i]]
            res[: len(distr), i] = distr

        distributions = res

    return distributions


def combine_spread_distributions(prev_spread_distributions, new_spread_distributions):
    distributions = {
        "N": prev_spread_distributions["N"] + new_spread_distributions["N"]
    }

    for side in constants.ASKS_AND_BIDS:
        distribution = np.zeros(
            max(
                len(prev_spread_distributions[side]),
                len(new_spread_distributions[side]),
            )
        )

        distribution[: len(prev_spread_distributions[side])] += (
            prev_spread_distributions[side] * prev_spread_distributions["N"]
        )
        distribution[: len(new_spread_distributions[side])] += (
            new_spread_distributions[side] * new_spread_distributions["N"]
        )
        distribution /= distributions["N"]

        distributions[side] = distribution

    return distributions


def save_spread_distributions(data_dir, debug=False):
    spreads_dir = os.path.join(data_dir, "spreads")
    if not os.path.exists(spreads_dir) and not debug:
        os.mkdir(spreads_dir)

    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    markets = get_filtered_markets(client, filter_volume=True)

    print("Saving average spread (orderbook) data...")

    max_N = 0
    max_symbol = None

    max_distribution_length = 0
    max_distr_symbol = None

    for coin, markets_group in tqdm(markets.groupby("coin")):
        coin_dir = os.path.join(spreads_dir, coin)
        if not os.path.exists(coin_dir) and not debug:
            os.mkdir(coin_dir)

        for market in markets_group["name"]:
            symbol = market.split("/")[0]

            prev_spread_distributions = load_spread_distributions(data_dir, symbol)

            new_spread_distributions = get_spread_distributions(client, market)

            spread_distributions = combine_spread_distributions(
                prev_spread_distributions, new_spread_distributions
            )

            if spread_distributions["N"] > max_N:
                max_N = spread_distributions["N"]
                max_symbol = symbol

            for side in constants.ASKS_AND_BIDS:
                if len(spread_distributions[side]) > max_distribution_length:
                    max_distribution_length = len(spread_distributions[side])
                    max_distr_symbol = symbol

            data_file = os.path.join(coin_dir, symbol + ".json")
            if not debug:
                with open(data_file, "w") as file:
                    json.dump(spread_distributions, file, cls=utils.NpEncoder)

    print(f"Done, max. effective number of orderbooks: {max_N} (symbol {max_symbol})")
    print(
        f"Max distribution size: {max_distribution_length} (symbol {max_distr_symbol})"
    )
    print()


def visualize_spreads(data_dir, symbol):
    distributions = load_spread_distributions(data_dir, symbol, stack=True)

    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    total_balance = utils.get_total_balance(client, False)
    total_balances = np.logspace(
        np.log10(total_balance / 10), np.log10(total_balance * 10), 100
    )
    spreads = np.zeros(len(total_balances))
    for i, balance in enumerate(total_balances):
        spreads[i] = utils.calculate_max_average_spread(distributions, balance)

    spread = utils.calculate_max_average_spread(distributions, total_balance)
    print(spread)
    plt.style.use("seaborn")
    plt.plot(total_balances, spreads)
    plt.plot([total_balance], [spread], ".k")
    plt.show()


def get_order_history(client, end_time):
    order_history = client.get_order_history(end_time=end_time)
    time.sleep(0.05)
    order_history = pd.DataFrame(order_history)
    return order_history


def get_conditional_order_history(client, end_time):
    conditional_order_history = client.get_conditional_order_history(end_time=end_time)
    time.sleep(0.05)
    conditional_order_history = pd.DataFrame(conditional_order_history)
    return conditional_order_history


def get_timestamps(time_string_series, time_string_format="%Y-%m-%dT%H:%M:%S"):
    timestamps = time_string_series.map(
        lambda x: time.mktime(
            datetime.strptime(
                x[: len(time_string_format) + 2], time_string_format
            ).timetuple()
        )
    )
    return timestamps


def get_trade_history(client, get_order_history_f, max_time=None):
    if max_time is None:
        max_time = -1

    end_time = time.time()
    order_history = get_order_history_f(client, end_time)
    timestamps = get_timestamps(order_history["createdAt"])
    end_time = timestamps.min()

    res = [order_history]

    while len(order_history.index) == 100 and end_time > max_time:
        order_history = get_order_history_f(client, end_time)
        res.append(order_history)
        timestamps = get_timestamps(order_history["createdAt"])
        end_time = timestamps.min()

    res = pd.concat(res, ignore_index=True)
    res = res[res["filledSize"] > 0.0]

    return res


def get_trades_data_frame_and_max_time(fname):
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col=0)
        max_time = get_timestamps(df["createdAt"]).max()
    else:
        df = None
        max_time = None

    return df, max_time


def save_trade_history(data_dir):
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    trades_fname = data_dir + "/" + "trades.csv"
    conditional_trades_fname = data_dir + "/" + "conditional_trades.csv"

    def _helper(fname, get_order_history_f):
        prev_trades, max_time = get_trades_data_frame_and_max_time(fname)
        n = len(prev_trades) if prev_trades is not None else 0
        trades = [prev_trades] if prev_trades is not None else []

        trade_history = get_trade_history(
            client, get_order_history_f, max_time=max_time
        )

        trades.append(trade_history)
        trades = (
            pd.concat(trades, ignore_index=True)
            .drop_duplicates("id")
            .sort_values("createdAt")
        )

        print(f"Saved {len(trades) - n} new trade(s) (total length {len(trades)})")

        trades.to_csv(fname)

    _helper(trades_fname, get_order_history)
    _helper(conditional_trades_fname, get_conditional_order_history)

    print()


def save_total_balance(data_dir):
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    balance = pd.DataFrame(
        [
            [
                time.time(),
                utils.get_total_balance(client, separate=False),
            ]
        ],
        columns=["time", "balance"],
    )

    fname = data_dir + "/" + "balances.csv"
    balances = [balance]

    if os.path.exists(fname):
        balances.append(pd.read_csv(fname, index_col=0))

    balances = pd.concat(balances, ignore_index=True).sort_values("time")
    balances.to_csv(fname)

    print(f"Saved total balance (total length {len(balances)})")
    print()


def save_average_volumes(data_dir):
    print("Saving average volumes...")

    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    markets = get_filtered_markets(client)

    volumes = {}

    max_volume = -1
    max_symbol = None

    for market in tqdm(markets["name"]):
        price_data = load_price_data(data_dir, market, return_price_data_only=True)
        symbol = market.split("/")[0]

        volumes[symbol] = price_data["volume"].mean()

        if volumes[symbol] > max_volume:
            max_volume = volumes[symbol]
            max_symbol = symbol

    volumes_fname = os.path.abspath(os.path.join(data_dir, "volumes.json"))

    with open(volumes_fname, "w") as file:
        json.dump(volumes, file, cls=utils.NpEncoder)

    print(f"Largest average volume: {max_volume} (symbol: {max_symbol})")
    print()


def experiments():
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    X = get_price_data(client, "PERP/USD", limit=None, prev_end_time=None, verbose=True)

    print(X)
    print(len(X))
    X["t"] = X["startTime"].map(lambda x: datetime.timestamp(parse_datetime(x)))
    print(np.unique(np.diff(X["t"])) / 60)


# TODO: remove HALF symbols
# TODO: remove "startTimestamp" and "index" columns
def clean(data_dir):
    pass


def main():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    save_price_data(data_dir)
    save_spread_distributions(data_dir)
    save_trade_history(data_dir)
    save_total_balance(data_dir)
    save_average_volumes(data_dir)
    clean(data_dir)

    # get_and_save_all(data_dir)
    # save_orderbook_data(data_dir)


# TODO: save deposits/withdrawals?
if __name__ == "__main__":
    main()
    # experiments()
