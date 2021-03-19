import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algtra.collect import data
from algtra import utils
from algtra import constants
from ftx.rest.client import FtxClient
import keys
from hypothesis import given, assume, example, strategies as st, settings
import numpy as np
import time
from datetime import datetime, timedelta
from ciso8601 import parse_datetime
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype


def test_get_filtered_markets():
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
    markets = data.get_filtered_markets(client)

    for curr in constants.NON_USD_FIAT_CURRENCIES:
        # TODO: change this to pd.testing somehow?
        assert not markets["name"].str.contains(curr).any()


class Test_get_price_data:
    def test_timestamps(self):
        client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

        prev_end_time = time.time()

        price_data = data.get_price_data(client, "ETH/USD", prev_end_time=prev_end_time)

        assert (
            time.time()
            - price_data["startTime"]
            .map(lambda x: datetime.timestamp(parse_datetime(x).now()))
            .max()
            < 60
        )
        assert (
            price_data["startTime"]
            .map(lambda x: datetime.timestamp(parse_datetime(x).now()))
            .min()
            > prev_end_time
        )

    @settings(max_examples=2, deadline=None)
    @given(limit=st.integers(1, 10000))
    def test_limit(self, limit):
        client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

        price_data = data.get_price_data(client, "BTC/USD", limit=limit)

        assert len(price_data) == limit


@given(symbol=st.text(min_size=1))
@example(symbol="ETH")
@example(symbol="BULL")
def test_load_price_data(symbol):
    assume("\\" not in symbol)
    assume("\x00" not in symbol)
    assume("/" not in symbol)
    file_location = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(file_location, "../data"))
    market = symbol + "/USD"
    coin = utils.get_coin(symbol)

    assert (
        data.load_price_data(data_dir, market, return_price_data=False) is None
    ) == (not os.path.exists(os.path.join(data_dir, coin)))


def test_get_spread_distributions():
    client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    distributions = data.get_spread_distributions(client, "FTT/USD")

    assert distributions["N"] == 1
    for side in constants.ASKS_AND_BIDS:
        assert side in distributions
        assert len(distributions[side] > 2)


@given(symbol=st.text(min_size=1))
@example(symbol="ETH")
@example(symbol="BULL")
def test_load_spread_distributions(symbol):
    assume("\\" not in symbol)
    assume("\x00" not in symbol)
    assume("/" not in symbol)
    file_location = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(os.path.join(file_location, "../data"))
    coin = utils.get_coin(symbol)

    distributions = data.load_spread_distributions(data_dir, symbol)

    assert (distributions["N"] == 0) == (
        not os.path.exists(os.path.join(data_dir, coin))
    )
    for side in constants.ASKS_AND_BIDS:
        assert (len(distributions[side]) == 1) == (
            not os.path.exists(os.path.join(data_dir, "spreads", coin))
        )


@given(
    prev_N=st.integers(1, 1e4),
    new_N=st.integers(1, 1e4),
    prev_distribution=arrays(
        np.float64,
        st.integers(1, 100),
        elements=st.floats(-1e7, 1e7, width=64),
    ),
    new_distribution=arrays(
        np.float64,
        st.integers(1, 100),
        elements=st.floats(-1e7, 1e7, width=64),
    ),
)
def test_combine_spread_distributions(
    prev_N, new_N, prev_distribution, new_distribution
):
    prev_distributions = {"N": prev_N}
    new_distributions = {"N": new_N}

    for side in constants.ASKS_AND_BIDS:
        prev_distributions[side] = prev_distribution
        new_distributions[side] = new_distribution

    distributions = data.combine_spread_distributions(
        prev_distributions, new_distributions
    )

    padded_prev_distribution = np.concatenate(
        [
            prev_distribution,
            np.zeros(max(0, len(new_distribution) - len(prev_distribution))),
        ]
    )
    padded_new_distribution = np.concatenate(
        [
            new_distribution,
            np.zeros(max(0, len(prev_distribution) - len(new_distribution))),
        ]
    )

    assert distributions["N"] == prev_N + new_N
    for side in constants.ASKS_AND_BIDS:
        assert len(distributions[side]) == max(
            len(prev_distribution), len(new_distribution)
        )

        np.testing.assert_almost_equal(
            distributions[side],
            np.average(
                np.stack([padded_prev_distribution, padded_new_distribution]),
                axis=0,
                weights=np.array([prev_N, new_N]),
            ),
        )
