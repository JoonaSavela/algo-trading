from algtra import data
from algtra import constants
import pandas as pd
from hypothesis import given, assume, example, strategies as st, settings
import pytest

# import os
# import sys

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from algtra import utils
# import numpy as np
# import time
# from datetime import datetime, timedelta
# from ciso8601 import parse_datetime
# from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype
# import glob


def test_PriceDataProcessor_fetch_with_invalid_input_should_fail(pdp):
    with pytest.raises(TypeError) as e:
        pdp.fetch()

    with pytest.raises(ValueError) as e:
        pdp.fetch("")

    with pytest.raises(ValueError) as e:
        pdp.fetch("asdf")

    with pytest.raises(ValueError) as e:
        pdp.fetch("BTC", -1)

    with pytest.raises(ValueError) as e:
        pdp.fetch("BTC", 0)


def test_PriceDataProcessor_fetch_with_valid_input_should_return_pd_DataFrame(
    pdp,
):
    assert isinstance(pdp.fetch("BTC", 10), pd.DataFrame)


@given(limit=st.integers(1, 9999))
@settings(max_examples=2, deadline=None)
def test_PriceDataProcessor_fetch_should_return_pd_DataFrame_with_length_equal_to_limit(
    limit,
):
    pdp = data.PriceDataProcessor()

    price_data = pdp.fetch("BTC", limit)

    assert len(price_data == limit)


def test_PriceDataProcessor_fetch_should_return_pd_DataFrame_with_specific_columns(pdp):
    assert set(pdp.fetch("BTC", 10).columns) == {
        "close",
        "open",
        "high",
        "low",
        "volume",
        "time",
    }


# def test_get_filtered_markets():
#     client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
#     markets = data.get_filtered_markets(client)

#     for curr in constants.NON_USD_FIAT_CURRENCIES:
#         # TODO: change this to pd.testing somehow?
#         assert not markets["name"].str.contains(curr).any()


# class Test_get_price_data:
#     @pytest.mark.slow
#     @given(minutes=st.integers(1, 1000))
#     @settings(max_examples=2, deadline=None)
#     def test_timestamps(self, minutes):
#         client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

#         prev_end_time = time.time() - 60 * minutes

#         price_data = data.get_price_data(client, "ETH/USD", prev_end_time=prev_end_time)

#         if len(price_data) > 0:
#             price_data["time"] = price_data["startTime"].map(
#                 lambda x: datetime.timestamp(parse_datetime(x))
#             )

#             assert time.time() - price_data["time"].max() < 60 + 40
#             assert price_data["time"].min() > prev_end_time
#             assert price_data["time"].min() - prev_end_time < 60

#     @pytest.mark.slow
#     @settings(max_examples=2, deadline=None)
#     @given(limit=st.integers(1, 10000))
#     def test_limit(self, limit):
#         client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

#         price_data = data.get_price_data(client, "BTC/USD", limit=limit)

#         assert len(price_data) == limit


# def test_glob_paths():
#     for x in glob.glob(
#         constants.DATA_STORAGE_LOCATION + "/" + constants.LOCAL_DATA_DIR + "/*/*.csv"
#     ):
#         path_end = x.split(
#             constants.DATA_STORAGE_LOCATION + "/" + constants.LOCAL_DATA_DIR
#         )[1]

#         assert "spreads" not in path_end
#         assert "balances" not in path_end
#         assert "trades" not in path_end
#         assert "conditional_trades" not in path_end
#         assert "volumes" not in path_end


# class Test_load_price_data:
#     @pytest.mark.slow
#     @given(
#         fname=st.sampled_from(
#             glob.glob(
#                 constants.DATA_STORAGE_LOCATION
#                 + "/"
#                 + constants.LOCAL_DATA_DIR
#                 + "/*/*.csv"
#             )
#         )
#     )
#     @settings(max_examples=2, deadline=None)
#     def test_return_price_data_False(self, fname):
#         fname_split = fname.split("/")
#         symbol = fname_split[-1].split("_")[0]
#         market = symbol + "/USD"
#         coin = utils.get_coin(symbol)

#         data_dir = fname.split(coin)[0]

#         price_data, prev_end_time1, data_length1 = data.load_price_data(
#             data_dir, market, return_price_data=True
#         )
#         prev_end_time2, data_length2 = data.load_price_data(
#             data_dir, market, return_price_data=False
#         )

#         assert price_data["time"].max() == prev_end_time1
#         assert prev_end_time1 == prev_end_time2
#         assert data_length1 == data_length2

#     @pytest.mark.slow
#     @given(symbol=st.text(min_size=1))
#     @example(symbol="ETH")
#     @example(symbol="BULL")
#     @settings(max_examples=2, deadline=None)
#     def test_existance_and_nonexistance(self, symbol):
#         assume("\\" not in symbol)
#         assume("\x00" not in symbol)
#         assume("/" not in symbol)
#         data_dir = os.path.abspath(
#             os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
#         )
#         market = symbol + "/USD"
#         coin = utils.get_coin(symbol)

#         prev_end_time, _ = data.load_price_data(
#             data_dir, market, return_price_data=False
#         )

#         assert (prev_end_time is None) == (
#             not os.path.exists(os.path.join(data_dir, coin))
#         )


# @given(
#     prices=arrays(
#         np.float64,
#         st.integers(2, 200),
#         elements=st.floats(0.001, 1e7, width=64),
#     ),
#     sizes=arrays(
#         np.float64,
#         st.integers(2, 200),
#         elements=st.floats(0.001, 1e6, width=64),
#     ),
# )
# def test_clean_orderbook(prices, sizes):
#     assume(len(prices) <= len(sizes))
#     sizes = sizes[: len(prices)]

#     N = len(prices) // 2

#     prices1, prices2 = list(prices[:N]), list(prices[N:])
#     sizes1, sizes2 = sizes[:N], sizes[N:]

#     prices1 = sorted(prices1, reverse=True)
#     prices2 = sorted(prices2)

#     orderbook = {
#         "bids": [[price, size] for price, size in zip(prices1, sizes1)],
#         "asks": [[price, size] for price, size in zip(prices2, sizes2)],
#     }

#     assert all(
#         orderbook["bids"][i][0] >= orderbook["bids"][i + 1][0]
#         for i in range(len(orderbook["bids"]) - 1)
#     )
#     assert all(
#         orderbook["asks"][i][0] <= orderbook["asks"][i + 1][0]
#         for i in range(len(orderbook["asks"]) - 1)
#     )

#     orderbook_size_diff1 = data.orderbook_size_diff(orderbook)
#     cleaned_orderbook = data.clean_orderbook(orderbook)
#     orderbook_size_diff2 = data.orderbook_size_diff(orderbook)

#     np.testing.assert_almost_equal(orderbook_size_diff1, orderbook_size_diff2)

#     orderbook = cleaned_orderbook

#     assert all(
#         orderbook["bids"][i][0] >= orderbook["bids"][i + 1][0]
#         for i in range(len(orderbook["bids"]) - 1)
#     )
#     assert all(
#         orderbook["asks"][i][0] <= orderbook["asks"][i + 1][0]
#         for i in range(len(orderbook["asks"]) - 1)
#     )

#     if orderbook["bids"] and orderbook["asks"]:
#         assert orderbook["bids"][0][0] <= orderbook["asks"][0][0]


# def test_get_spread_distributions():
#     client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

#     distributions = data.get_spread_distributions(client, "FTT/USD")

#     assert distributions["N"] == 1
#     for side in constants.ASKS_AND_BIDS:
#         assert side in distributions
#         assert len(distributions[side] > 2)


# @given(symbol=st.text(min_size=1))
# @example(symbol="ETH")
# @example(symbol="BULL")
# @settings(deadline=timedelta(milliseconds=1000))
# def test_load_spread_distributions(symbol):
#     assume("\\" not in symbol)
#     assume("\x00" not in symbol)
#     assume("/" not in symbol)
#     assume("." not in symbol)
#     data_dir = os.path.abspath(
#         os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
#     )
#     coin = utils.get_coin(symbol)

#     distributions = data.load_spread_distributions(data_dir, symbol)

#     assert (distributions["N"] == 0) == (
#         not os.path.exists(os.path.join(data_dir, coin))
#     )
#     for side in constants.ASKS_AND_BIDS:
#         assert (len(distributions[side]) == 1) == (
#             not os.path.exists(os.path.join(data_dir, "spreads", coin))
#         )

#     distributions = data.load_spread_distributions(data_dir, symbol, stack=True)

#     if os.path.exists(os.path.join(data_dir, coin)):
#         assert isinstance(distributions, np.ndarray)
#         assert distributions.shape[1] == 2


# @given(
#     prev_N=st.integers(1, 1e4),
#     new_N=st.integers(1, 1e4),
#     prev_distribution=arrays(
#         np.float64,
#         st.integers(1, 100),
#         elements=st.floats(-1e7, 1e7, width=64),
#     ),
#     new_distribution=arrays(
#         np.float64,
#         st.integers(1, 100),
#         elements=st.floats(-1e7, 1e7, width=64),
#     ),
# )
# def test_combine_spread_distributions(
#     prev_N, new_N, prev_distribution, new_distribution
# ):
#     prev_distributions = {"N": prev_N}
#     new_distributions = {"N": new_N}

#     for side in constants.ASKS_AND_BIDS:
#         prev_distributions[side] = prev_distribution
#         new_distributions[side] = new_distribution

#     distributions = data.combine_spread_distributions(
#         prev_distributions, new_distributions
#     )

#     padded_prev_distribution = np.concatenate(
#         [
#             prev_distribution,
#             np.zeros(max(0, len(new_distribution) - len(prev_distribution))),
#         ]
#     )
#     padded_new_distribution = np.concatenate(
#         [
#             new_distribution,
#             np.zeros(max(0, len(prev_distribution) - len(new_distribution))),
#         ]
#     )

#     assert distributions["N"] == prev_N + new_N
#     for side in constants.ASKS_AND_BIDS:
#         assert len(distributions[side]) == max(
#             len(prev_distribution), len(new_distribution)
#         )

#         np.testing.assert_almost_equal(
#             distributions[side],
#             np.average(
#                 np.stack([padded_prev_distribution, padded_new_distribution]),
#                 axis=0,
#                 weights=np.array([prev_N, new_N]),
#             ),
#         )


# def test_split_price_data():
#     data_dir = os.path.abspath(
#         os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
#     )

#     market = "FTT/USD"

#     price_data = data.load_price_data(data_dir, market, return_price_data_only=True)
#     price_data_splits = data.split_price_data(price_data)

#     for price_data in price_data_splits:
#         assert len(price_data) > constants.MIN_AGGREGATE_N * constants.MIN_W * 60

#         time_diffs = np.diff(price_data["time"].values) // 60

#         assert np.all(time_diffs <= constants.PRICE_DATA_MAX_GAP)
