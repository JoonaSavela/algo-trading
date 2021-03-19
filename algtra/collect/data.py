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
from algtra.constants import NON_USD_FIAT_CURRENCIES, ORDERBOOK_STEP_SIZE, ASKS_AND_BIDS
import time
from ciso8601 import parse_datetime
from tqdm import tqdm
from pathlib import Path
from loguru import logger


def _get_recent_data(coin, TimeTo, size, type, aggregate):
    if type == "m":
        url = "https://min-api.cryptocompare.com/data/histominute"
    elif type == "h":
        url = "https://min-api.cryptocompare.com/data/histohour"
    else:
        raise ValueError('type must be "m" or "h"')
    url += (
        "?fsym="
        + coin
        + "&tsym=USD&limit="
        + str(size - 1)
        + ("&toTs={}".format(TimeTo) if TimeTo is not None else "")
        + "&aggregate="
        + str(aggregate)
        + "&api_key="
        + keys.cryptocompare_key
    )

    request = requests.get(url)
    content = request.json()

    data = content["Data"]
    data_keys = ["close", "high", "low", "open", "volumefrom", "volumeto"]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key in data_keys:
            tmp.append(item[key])
        X[i, :] = tmp

    if content["Response"] == "Error":
        print(content)

    return X, content["TimeTo"], content["TimeFrom"]


def get_recent_data(coin, size=3 * 14, type="m", aggregate=1):
    res = np.zeros(shape=(size, 6))

    s = size
    timeFrom = None

    while s > 0:
        if s == size:
            X, timeTo, timeFrom = _get_recent_data(
                coin, timeFrom, min(2000, s), type, aggregate
            )
        else:
            X, _, timeFrom = _get_recent_data(
                coin, timeFrom, min(2000, s), type, aggregate
            )

        s -= X.shape[0]

        res[s : s + X.shape[0], :] = X

    return res, timeTo


def get_and_save(data_dir, coin, t):
    if t > time.time():
        return False
    time_str = str(t)
    url = (
        "https://min-api.cryptocompare.com/data/histominute?fsym="
        + coin
        + "&tsym=USD&limit=2000&toTs="
        + time_str
        + "&api_key="
        + keys.cryptocompare_key
    )
    request = requests.get(url)
    content = json.loads(request._content)

    is_same_time = content["TimeTo"] == int(time_str)

    if (
        content["Response"] == "Success"
        and len(content["Data"]) > 2000
        and is_same_time
    ):
        with open(data_dir + "/" + coin + "/" + time_str + ".json", "w") as file:
            json.dump(content, file)
    elif not is_same_time:
        # print('The "To" time was different than expected')
        pass
    elif len(content["Data"]) <= 2000:
        print("Data length is under 2001")
    else:
        print("An error occurred")
        for key, item in content.items():
            print(key, item)
        print()

    return is_same_time


# TODO: move this somewhere more sensible
coins = ["BTC", "ETH", "XRP", "BCH", "LTC"]


def get_and_save_all(data_dir):
    for coin in coins:
        if not os.path.exists(data_dir + "/" + coin):
            os.mkdir(data_dir + "/" + coin)
        t_max = -1
        for filename in glob.glob(data_dir + "/" + coin + "/*.json"):
            split1 = filename.split("/")
            split2 = split1[-1].split(".")
            if int(split2[0]) > t_max:
                t_max = int(split2[0])

        if t_max != -1:
            t = t_max + 2000 * 60
        else:
            url = (
                "https://min-api.cryptocompare.com/data/histominute?fsym="
                + coin
                + "&tsym=USD&limit=2000&api_key="
                + keys.cryptocompare_key
            )
            request = requests.get(url)
            content = json.loads(request._content)
            t = content["TimeTo"] - 7 * 24 * 60 * 60 + 2000 * 60
            print(coin + ": No previous files found")

        count = 0

        while get_and_save(data_dir, coin, t):
            t += 2000 * 60
            count += 1

        print("Coin", coin, "processed,", count, "interval(s) added")
    print()


# TODO: use pandas
def load_data(filename, sequence_length):
    obj = {}

    with open(filename, "r") as file:
        obj = json.load(file)

    data = obj["Data"][:sequence_length]

    data_keys = ["close", "high", "low", "open", "volumefrom", "volumeto"]

    X = np.zeros(shape=(len(data), 6))
    for i in range(len(data)):
        item = data[i]
        tmp = []
        for key in data_keys:
            tmp.append(item[key])
        X[i, :] = tmp

    return X


def load_all_data(filenames, index=0, return_time=False):
    filenames = sorted(filenames, key=utils.get_time)

    idx = np.arange(1, len(filenames))
    li = np.diff(np.array(list(map(lambda x: utils.get_time(x), filenames)))) != 120000

    points = [0]
    for p in idx[li]:
        points.append(p)
    points.append(len(filenames))

    points = list(zip(points[:-1], points[1:]))

    if isinstance(index, int):
        # TODO: change this into an assertion
        index = min(index, len(points) - 1)
        idx = [index]
    elif not isinstance(index, list):
        raise ValueError("index must be either int or list")
    else:
        idx = index

    res = []

    for start, end in map(lambda x: points[x], idx):
        fnames = filenames[start:end]

        Xs = []

        for filename in fnames:
            X = load_data(filename, 2001)
            Xs.append(X[:2000, :])  # remove duplicates

        X = np.concatenate(Xs)
        res.append(X)

    if len(res) == 1:
        if return_time:
            return res[0], utils.get_time(filenames[points[idx[0]][1] - 1])
        return res[0]

    if return_time:
        return res, list(
            map(lambda x: utils.get_time(filenames[points[x][1] - 1]), idx)
        )
    return res


def save_orderbook_data_old(data_dir):
    source_symbol = "USD"
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    for coin in coins:
        if not os.path.exists(data_dir + "/" + "orderbooks/" + coin):
            os.mkdir(data_dir + "/" + "orderbooks/" + coin)

        symbols = [
            coin,
            "BULL" if coin == "BTC" else coin + "BULL",
            "BEAR" if coin == "BTC" else coin + "BEAR",
            "HEDGE" if coin == "BTC" else coin + "HEDGE",
        ]

        for symbol in symbols:
            if not os.path.exists(data_dir + "/" + "orderbooks/" + coin + "/" + symbol):
                os.mkdir(data_dir + "/" + "orderbooks/" + coin + "/" + symbol)
            orderbook = client.get_orderbook(symbol + "/" + source_symbol, 100)
            time.sleep(0.05)

            filename = (
                data_dir
                + "/"
                + "orderbooks/"
                + coin
                + "/"
                + symbol
                + "/"
                + str(round(time.time()))
                + ".json"
            )
            with open(filename, "w") as fp:
                json.dump(orderbook, fp)

        print(coin, "orderbook data saved")
    print()


def get_spread_distributions(client, market):
    orderbook = client.get_orderbook(market, 100)
    time.sleep(0.05)

    asks = np.array(orderbook["asks"])
    bids = np.array(orderbook["bids"])
    side_arrays = {"asks": asks, "bids": bids}
    price = (asks[0, 0] + bids[0, 0]) / 2

    distributions = {"N": 1}

    for side in ASKS_AND_BIDS:
        usd_values = np.prod(orderbook[side], axis=1)
        log_percentages = np.log(side_arrays[side][:, 0] / price)

        try:
            assert np.all(np.diff(np.abs(log_percentages)) >= 0)
        except AssertionError:
            logger.error(log_percentages.tostring())
            raise

        distribution = np.zeros(
            int(np.max(np.abs(log_percentages)) / ORDERBOOK_STEP_SIZE + 2)
        )
        idx = np.ceil(np.abs(log_percentages) / ORDERBOOK_STEP_SIZE).astype(int)

        for i in range(len(idx)):
            distribution[idx[i]] += usd_values[i]

        distributions[side] = distribution

    return distributions


def load_spread_distributions(data_dir, symbol):
    coin = utils.get_coin(symbol)
    coin_dir = os.path.join(data_dir, "spreads", coin)

    data_file = os.path.join(coin_dir, symbol + ".json")

    # Default if file doesn't exist
    distributions = {"N": 0}
    for side in ASKS_AND_BIDS:
        distributions[side] = np.array([0])

    if os.path.exists(data_file):
        with open(data_file, "r") as file:
            distributions = json.load(file)

        for side in ASKS_AND_BIDS:
            distributions[side] = np.array(distributions[side])

    return distributions


def combine_spread_distributions(prev_spread_distributions, new_spread_distributions):
    distributions = {
        "N": prev_spread_distributions["N"] + new_spread_distributions["N"]
    }

    for side in ASKS_AND_BIDS:
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

            for side in ASKS_AND_BIDS:
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


def calculate_max_average_spread(distributions, total_balance):
    res = []

    for side in ASKS_AND_BIDS:
        li = np.cumsum(distributions[side]) < total_balance
        weights = distributions[side][li] / total_balance

        if len(weights) < len(distributions[side]):
            weights = np.concatenate([weights, [1 - np.sum(weights)]])
        else:
            weights[-1] += 1 - np.sum(weights)

        log_percentage = np.arange(len(distributions[side])) * ORDERBOOK_STEP_SIZE

        average_spread = np.dot(log_percentage[: len(weights)], weights)

    return np.exp(np.max(res)) - 1


def visualize_spreads(coin="ETH", m=1, m_bear=1):
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    total_balance = utils.get_total_balance(client, False)
    total_balances = np.logspace(
        np.log10(total_balance / 10), np.log10(total_balance * 100), 100
    )
    spreads = get_average_spread(coin, m, total_balances, m_bear)
    spread = spreads[np.argmin(np.abs(total_balances - total_balance))]
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


def get_filtered_markets(client, filter_volume=False):
    markets = pd.DataFrame(client.list_markets())
    time.sleep(0.05)

    markets = markets[markets["name"].str.contains("/USD")]
    markets = markets[markets["quoteCurrency"] == "USD"]
    markets = markets[~markets["name"].str.contains("USDT")]
    for curr in NON_USD_FIAT_CURRENCIES:
        markets = markets[~markets["name"].str.contains(curr)]
    markets = markets[markets["tokenizedEquity"].isna()]
    if filter_volume:
        markets = markets[markets["volumeUsd24h"] > 0]

    markets["coin"] = markets["baseCurrency"].map(lambda x: utils.get_coin(x))
    markets = markets.sort_values("volumeUsd24h", ascending=False)

    return markets


# TODO: intead of limit=5000, calculate the optimal limit based on prev_end_time
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
        len(X) >= 5000
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
            res["startTime"].map(lambda x: datetime.timestamp(parse_datetime(x).now()))
            > prev_end_time
        ]

    return res


def load_price_data(
    data_dir, market, return_price_data=True, return_price_data_only=False
):
    symbol = market.split("/")[0]
    coin = utils.get_coin(symbol)

    data_file = os.path.join(data_dir, coin, symbol + ".csv")
    price_data = None
    prev_end_time = None
    data_length = 0

    if os.path.exists(data_file):
        price_data = pd.read_csv(data_file, index_col=0)
        prev_end_time = datetime.timestamp(
            parse_datetime(price_data["startTime"].max()).now()
        )
        data_length = len(price_data)

    if return_price_data:
        if return_price_data_only:
            return price_data

        return price_data, prev_end_time, data_length

    return prev_end_time


def save_price_data(data_dir):
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    markets = get_filtered_markets(client)

    datapoints_added = 0
    total_datapoints = 0

    print("Saving price data...")

    for coin, markets_group in tqdm(markets.groupby("coin")):
        coin_dir = os.path.join(data_dir, coin)
        if not os.path.exists(coin_dir):
            os.mkdir(coin_dir)

        for market in markets_group["name"]:
            (
                prev_price_data,
                prev_end_time,
                prev_data_length,
            ) = load_price_data(data_dir, market, return_price_data=True)

            new_price_data = get_price_data(
                client,
                market,
                limit=None,
                prev_end_time=prev_end_time,
                verbose=False,
            )

            symbol = market.split("/")[0]
            if (
                new_price_data["startTime"]
                .map(lambda x: datetime.timestamp(parse_datetime(x).now()))
                .min()
                - prev_end_time
                > 60
            ):
                logger.warning(
                    f"There is a gap in previous and new price data for symbol {symbol}. Splitting required."
                )

            price_data = (
                pd.concat([prev_price_data, new_price_data], ignore_index=True)
                .drop_duplicates("startTime")
                .sort_values("startTime")
            )
            datapoints_added += len(price_data) - prev_data_length
            total_datapoints += len(price_data)

            data_file = os.path.join(coin_dir, symbol + ".csv")
            price_data.to_csv(data_file)

    print(f"Saved {datapoints_added} price data points (total {total_datapoints})")
    print()


def experiments():
    pass


def clean(data_dir):
    pass


def main():
    # TODO: change this to cwd?
    FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(FILE_ROOT, "../../data"))
    data_dir = os.getcwd()
    if Path(PROJECT_ROOT) in Path(data_dir).parents:
        raise ValueError(
            "Current working directory must not be inside any subdirectory of the project."
        )
    data_dir = os.path.abspath(os.path.join(data_dir, "data"))

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    save_price_data(data_dir)
    save_spread_distributions(data_dir)

    # get_and_save_all(data_dir)
    # save_orderbook_data(data_dir)
    save_trade_history(data_dir)
    save_total_balance(data_dir)


# TODO: save deposits/withdrawals?
if __name__ == "__main__":
    main()
    # experiments()
