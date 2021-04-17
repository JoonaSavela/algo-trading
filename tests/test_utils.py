import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from algtra import utils, constants
from algtra.collect import data
from ftx.rest.client import FtxClient
import keys
from hypothesis import given, assume, example, strategies as st, settings
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype
from datetime import timedelta
import pytest
import glob


class Test_calculate_max_average_spread:
    @pytest.mark.slow
    @given(
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        balance=st.floats(0, 1e6, width=64),
    )
    @settings(deadline=timedelta(seconds=2.5))
    def test_against_naive(self, distribution, balance):
        assume(np.sum(distribution) > 0)

        distributions1 = {
            "asks": distribution,
            "bids": distribution,
        }

        max_avg_spread1 = utils.calculate_max_average_spread_naive(
            distributions1, balance
        )

        distributions2 = np.stack([distribution, distribution], axis=1)

        max_avg_spread2 = utils.calculate_max_average_spread(distributions2, balance)

        np.testing.assert_allclose(max_avg_spread1, max_avg_spread2)

    @pytest.mark.slow
    @given(
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        balances=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e6, width=64),
        ),
    )
    @settings(deadline=timedelta(seconds=2.5))
    def test_is_increasing(self, distribution, balances):
        assume(np.sum(distribution) > 0)

        balances = np.sort(balances)

        distributions = np.stack([distribution, distribution], axis=1)

        spreads = np.zeros(len(balances))
        for i, balance in enumerate(balances):
            spreads[i] = utils.calculate_max_average_spread(distributions, balance)

        assert np.all(np.diff(spreads) >= 0)


class Test_get_average_buy_price:
    @given(
        buy_prices=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e6, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e5, width=64),
        ),
        sell_size=st.floats(0.001, 1e5),
    )
    def test_naive(self, buy_prices, buy_sizes, sell_size):
        assume(len(buy_prices) <= len(buy_sizes))
        buy_sizes = buy_sizes[: len(buy_prices)]

        original_size_diff = np.sum(buy_sizes) - sell_size
        assume(original_size_diff >= 0)

        avg_buy_price, buy_size_diffs = utils.get_average_buy_price_naive(
            buy_prices, buy_sizes, sell_size
        )

        np.testing.assert_allclose(sell_size, np.sum(buy_size_diffs))
        np.testing.assert_allclose(
            np.sum(buy_sizes - buy_size_diffs), original_size_diff
        )
        if avg_buy_price >= np.max(buy_prices):
            np.testing.assert_allclose(avg_buy_price, np.max(buy_prices))
        if avg_buy_price <= np.min(buy_prices):
            np.testing.assert_allclose(avg_buy_price, np.min(buy_prices))

    @given(
        buy_prices=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e6, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.0, 1e5, width=64),
        ),
        sell_size=st.floats(0.001, 1e5),
    )
    @settings(deadline=timedelta(milliseconds=500))
    def test_against_naive(self, buy_prices, buy_sizes, sell_size):
        assume(len(buy_prices) <= len(buy_sizes))
        buy_sizes = buy_sizes[: len(buy_prices)]

        original_size_diff = np.sum(buy_sizes) - sell_size
        assume(original_size_diff >= 0)

        avg_buy_price1, buy_size_diffs1 = utils.get_average_buy_price_naive(
            buy_prices, buy_sizes, sell_size
        )
        avg_buy_price2, buy_size_diffs2 = utils.get_average_buy_price(
            buy_prices, buy_sizes, sell_size
        )

        np.testing.assert_allclose(avg_buy_price1, avg_buy_price2)
        np.testing.assert_allclose(buy_size_diffs1, buy_size_diffs2)


class Test_apply_spread_commissions_and_taxes_to_value_change:
    @pytest.mark.slow
    @given(
        value_change_multiplier=st.floats(-1.0, 1.0),
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        buy_prices=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e3, width=64),
        ),
        sell_price=st.floats(1e-3, 1e3),
        balance=st.floats(0, 1e6, width=64),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        taxes=st.booleans(),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=timedelta(milliseconds=1000))
    def test_naive(
        self,
        value_change_multiplier,
        usd_value,
        asset_value,
        buy_prices,
        buy_sizes,
        sell_price,
        balance,
        distribution,
        taxes,
        tax_exemption,
    ):
        if len(buy_sizes) <= len(buy_prices):
            buy_prices = buy_prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(buy_prices)]
        assume(np.sum(buy_sizes) > 1e-3)

        assume(usd_value + asset_value > 1e-3)

        if value_change_multiplier >= 0.0:
            value_change = value_change_multiplier * asset_value
        else:
            value_change = value_change_multiplier * usd_value
        assume(value_change <= np.sum(buy_sizes) * sell_price)
        assume(value_change >= -usd_value)

        orderbook_distributions = np.stack([distribution, distribution], axis=1)

        (
            new_usd_value,
            new_asset_value,
            new_buy_size,
            buy_size_diffs,
        ) = utils.apply_spread_commissions_and_taxes_to_value_change_naive(
            value_change,
            usd_value,
            asset_value,
            buy_prices,
            buy_sizes,
            sell_price,
            balance,
            orderbook_distributions,
            taxes,
            tax_exemption,
        )

        assert new_usd_value >= 0.0
        assert new_asset_value >= 0.0
        assert new_buy_size == 0.0 or np.all(buy_size_diffs == 0.0)

    @pytest.mark.slow
    @given(
        value_change_multiplier=st.floats(-1.0, 1.0),
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        buy_prices=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e3, width=64),
        ),
        sell_price=st.floats(1e-3, 1e3),
        balance=st.floats(0, 1e6, width=64),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        taxes=st.booleans(),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=timedelta(milliseconds=1500))
    def test_against_naive(
        self,
        value_change_multiplier,
        usd_value,
        asset_value,
        buy_prices,
        buy_sizes,
        sell_price,
        balance,
        distribution,
        taxes,
        tax_exemption,
    ):
        if len(buy_sizes) <= len(buy_prices):
            buy_prices = buy_prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(buy_prices)]
        assume(np.sum(buy_sizes) > 1e-3)

        assume(usd_value + asset_value > 1e-3)

        if value_change_multiplier >= 0.0:
            value_change = value_change_multiplier * asset_value
        else:
            value_change = value_change_multiplier * usd_value
        assume(value_change <= np.sum(buy_sizes) * sell_price)
        assume(value_change >= -usd_value)

        orderbook_distributions = np.stack([distribution, distribution], axis=1)

        (
            new_usd_value1,
            new_asset_value1,
            new_buy_size1,
            buy_size_diffs1,
        ) = utils.apply_spread_commissions_and_taxes_to_value_change_naive(
            value_change,
            usd_value,
            asset_value,
            buy_prices,
            buy_sizes,
            sell_price,
            balance,
            orderbook_distributions,
            taxes,
            tax_exemption,
        )

        (
            new_usd_value2,
            new_asset_value2,
            new_buy_size2,
            buy_size_diffs2,
        ) = utils.apply_spread_commissions_and_taxes_to_value_change(
            value_change,
            usd_value,
            asset_value,
            buy_prices,
            buy_sizes,
            sell_price,
            balance,
            orderbook_distributions,
            taxes,
            tax_exemption,
        )

        np.testing.assert_almost_equal(new_usd_value1, new_usd_value2)
        np.testing.assert_almost_equal(new_asset_value1, new_asset_value2)
        np.testing.assert_almost_equal(new_buy_size1, new_buy_size2)
        np.testing.assert_almost_equal(buy_size_diffs1, buy_size_diffs2)

    @pytest.mark.slow
    @given(
        usd_value=st.floats(0, 1.0),
        buy_prices=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e3, width=64),
        ),
        sell_price=st.floats(1e-3, 1e3),
        balance=st.floats(0, 1e6, width=64),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        taxes=st.booleans(),
        tax_exemption=st.booleans(),
    )
    @settings(max_examples=200, deadline=timedelta(milliseconds=1000))
    def test_new_usd_percentage_is_increasing_as_a_function_of_value_change(
        self,
        usd_value,
        buy_prices,
        buy_sizes,
        sell_price,
        balance,
        distribution,
        taxes,
        tax_exemption,
    ):
        if len(buy_sizes) <= len(buy_prices):
            buy_prices = buy_prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(buy_prices)]
        assume(np.sum(buy_sizes) > 1e-3)

        orderbook_distributions = np.stack([distribution, distribution], axis=1)

        sell_sizes = np.linspace(-usd_value / sell_price, np.sum(buy_sizes), 1000)
        asset_value = sell_sizes[-1] * sell_price
        assume(usd_value + asset_value > 1e-3)

        value_changes = sell_sizes * sell_price
        new_usd_percentages = np.zeros(len(sell_sizes))

        for i in range(len(sell_sizes)):
            new_usd_percentages[i] = utils.get_usd_percentage_for_value_change(
                value_changes[i],
                usd_value,
                asset_value,
                buy_prices,
                buy_sizes,
                sell_price,
                balance,
                orderbook_distributions,
                taxes,
                tax_exemption,
            )

        # TODO: calculate the new lower bound (taking spreads etc. into account)
        # avg_buy_price_diffs = np.diff(avg_buy_prices) / np.diff(value_changes)
        #
        # asset_values = asset_values[:-1]
        # taxes = taxes[:-1]
        # value_changes = value_changes[:-1]
        # total_values = total_values[:-1]
        #
        # np.testing.assert_array_less(0.0, value_changes)
        # np.testing.assert_array_less(
        #     0.0, asset_values * constants.TAX_RATE * value_changes
        # )
        #
        # # obtained from calculating the derivative of new_usd_percentages
        # # (w.r.t. value_changes) and solving when is it positive (w.r.t.
        # # avg_buy_price_diffs as a function of value_changes)
        # lower_bounds = (asset_values * taxes / value_changes - total_values) / (
        #     asset_values * constants.TAX_RATE * value_changes / sell_price
        # )
        # # => new_usd_percentages is increasing as a function of value_changes
        # # when this lower bound holds for avg_buy_price_diffs
        #
        # np.testing.assert_array_less(lower_bounds, avg_buy_price_diffs)
        # print(new_usd_percentages)
        np.testing.assert_array_less(0.0, np.diff(new_usd_percentages))


@pytest.mark.slow
@given(
    usd_value=st.floats(0, 1.0),
    asset_value=st.floats(0, 1.0),
    target_usd_percentage=st.floats(0, 1.0),
    buy_prices=arrays(
        np.float64,
        st.integers(1, 10),
        elements=st.floats(1e-3, 1e3, width=64),
    ),
    buy_sizes=arrays(
        np.float64,
        st.integers(1, 10),
        elements=st.floats(1e-3, 1e3, width=64),
    ),
    sell_price=st.floats(1e-3, 1e3),
    balance=st.floats(0, 1e6),
    distribution=arrays(
        np.float64,
        st.integers(1, 100),
        elements=st.floats(0.001, 1e7, width=64),
    ),
    tax_exemption=st.booleans(),
)
@settings(deadline=None)
def test_get_balanced_usd_and_asset_values(
    usd_value,
    asset_value,
    target_usd_percentage,
    buy_prices,
    buy_sizes,
    sell_price,
    balance,
    distribution,
    tax_exemption,
):
    if len(buy_sizes) <= len(buy_prices):
        buy_prices = buy_prices[: len(buy_sizes)]
    else:
        buy_sizes = buy_sizes[: len(buy_prices)]
    assume(np.sum(buy_sizes) > 1e-3)

    assume(usd_value + asset_value > 1e-3)

    orderbook_distributions = np.stack([distribution, distribution], axis=1)

    asset_value = min(asset_value, np.sum(buy_sizes) * sell_price)

    (
        usd_value_base_case,
        asset_value_base_case,
        new_buy_size,
        buy_size_diffs,
    ) = utils.get_balanced_usd_and_asset_values(
        usd_value,
        asset_value,
        target_usd_percentage,
        buy_prices,
        buy_sizes,
        sell_price,
        0.0,
        orderbook_distributions,
        taxes=False,
    )

    assert usd_value_base_case >= 0.0
    assert asset_value_base_case >= 0.0
    np.testing.assert_allclose(
        usd_value_base_case + asset_value_base_case, usd_value + asset_value
    )
    np.testing.assert_almost_equal(
        usd_value_base_case / (usd_value_base_case + asset_value_base_case),
        target_usd_percentage,
    )

    (
        usd_value_with_spread,
        asset_value_with_spread,
        new_buy_size,
        buy_size_diffs,
    ) = utils.get_balanced_usd_and_asset_values(
        usd_value,
        asset_value,
        target_usd_percentage,
        buy_prices,
        buy_sizes,
        sell_price,
        balance,
        orderbook_distributions,
        taxes=False,
    )

    assert usd_value_with_spread >= 0.0
    assert asset_value_with_spread >= 0.0
    np.testing.assert_almost_equal(
        usd_value_with_spread / (usd_value_with_spread + asset_value_with_spread),
        target_usd_percentage,
    )
    assert (
        usd_value_with_spread + asset_value_with_spread <= usd_value + asset_value
    ) or np.allclose(
        usd_value_with_spread + asset_value_with_spread, usd_value + asset_value
    )

    (
        usd_value_full_case,
        asset_value_full_case,
        new_buy_size,
        buy_size_diffs,
    ) = utils.get_balanced_usd_and_asset_values(
        usd_value,
        asset_value,
        target_usd_percentage,
        buy_prices,
        buy_sizes,
        sell_price,
        balance,
        orderbook_distributions,
        taxes=True,
        tax_exemption=tax_exemption,
    )

    assert usd_value_full_case >= 0.0
    assert asset_value_full_case >= 0.0
    np.testing.assert_almost_equal(
        usd_value_full_case / (usd_value_full_case + asset_value_full_case),
        target_usd_percentage,
    )
    if not tax_exemption:
        assert (
            usd_value_full_case + asset_value_full_case
            <= usd_value_with_spread + asset_value_with_spread
        )


# @given(logit=st.floats(-20, 20))
# def test_roundtrip_logit_to_p_p_to_logit(logit):
#     value0 = utils.logit_to_p(logit=logit)
#     value1 = utils.p_to_logit(p=value0)
#     np.testing.assert_allclose(logit, value1)


class Test_get_symbol_and_get_coin:
    def test_get_coin(self):
        client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
        markets = data.get_filtered_markets(client)

        for curr in markets["baseCurrency"]:
            coin = utils.get_coin(curr)
            for partial_leveraged_symbol in constants.PARTIAL_LEVERAGED_SYMBOLS:
                assert partial_leveraged_symbol not in coin

    @given(s=st.text(min_size=1))
    @example("BULL")
    @example("BEAR")
    @example("HEDGE")
    @example("HALF")
    def test_get_coin_is_fixed_point(self, s):
        assume(s is not None)

        s = utils.get_coin(s)
        assert utils.get_coin(s) == s

    @given(s=st.text(min_size=1), m=st.sampled_from([1, -1, 3, -3]))
    @example(s="BTC", m=1)
    @example(s="BTC", m=-1)
    @example(s="BTC", m=3)
    @example(s="BTC", m=-3)
    def test_roundtrip_get_symbol_get_coin(self, s, m):
        s0 = utils.get_symbol(s, m=np.abs(m), bear=m < 0)
        assert s0 != s or m == 1
        s1 = utils.get_coin(s0)
        assert s1 == s


@pytest.mark.slow
@given(
    fname=st.sampled_from(
        glob.glob(
            constants.DATA_STORAGE_LOCATION
            + "/"
            + constants.LOCAL_DATA_DIR
            + "/*/*.csv"
        )
    )
)
@settings(max_examples=1, deadline=None)
def test_get_displacements(fname):
    fname_split = fname.split("/")
    symbol = fname_split[-1].split("_")[0]
    market = symbol + "/USD"
    coin = utils.get_coin(symbol)

    data_dir = fname.split(coin)[0]

    price_data = data.load_price_data(data_dir, market, return_price_data_only=True)

    displacements = utils.get_displacements(price_data)

    assert np.issubdtype(displacements[0].dtype, np.integer)
    assert np.all(0 <= displacements) and np.all(displacements < 60)


class Test_get_next_displacement_index:
    @given(
        displacements=arrays(
            np.int32,
            st.integers(1, 1000),
            elements=st.integers(0, 10000),
        ),
        start=st.integers(0, 1000),
        displacement=st.integers(0, 10000),
    )
    def test_naive(self, displacements, start, displacement):
        start = min(start, len(displacements) - 1)

        i = utils.get_next_displacement_index_naive(displacements, start, displacement)

        assert (
            i == len(displacements)
            or i == np.argwhere(displacements[start:] == displacement)[0] + start
        )

    @given(
        displacements=arrays(
            np.int32,
            st.integers(1, 1000),
            elements=st.integers(0, 10000),
        ),
        start=st.integers(0, 1000),
        displacement=st.integers(0, 10000),
    )
    def test_against_naive(self, displacements, start, displacement):
        start = min(start, len(displacements) - 1)

        i1 = utils.get_next_displacement_index_naive(displacements, start, displacement)
        i2 = utils.get_next_displacement_index(displacements, start, displacement)

        assert i1 == i2


class Test_aggregate_from_displacement:
    @given(
        fname=st.sampled_from(
            glob.glob(
                constants.DATA_STORAGE_LOCATION
                + "/"
                + constants.LOCAL_DATA_DIR
                + "/*/*.csv"
            )
        ),
        displacement=st.integers(0, 59),
    )
    @settings(max_examples=1, deadline=None)
    def test_naive(self, fname, displacement):
        fname_split = fname.split("/")
        symbol = fname_split[-1].split("_")[0]
        market = symbol + "/USD"
        coin = utils.get_coin(symbol)

        data_dir = fname.split(coin)[0]

        price_data = data.load_price_data(data_dir, market, return_price_data_only=True)

        closes = price_data["close"].values
        lows = price_data["low"].values
        highs = price_data["high"].values
        times = price_data["time"].values
        displacements = utils.get_displacements(price_data)

        (
            aggregated_closes,
            aggregated_lows,
            aggregated_highs,
            split_indices,
            aggregate_indices,
        ) = utils.aggregate_from_displacement_naive(
            closes, lows, highs, times, displacements, displacement
        )

        assert len(aggregated_closes) == len(aggregated_lows)
        assert len(aggregated_closes) == len(aggregated_highs)
        assert len(aggregated_closes) == len(aggregate_indices)

        assert (
            abs(len(closes) // 60 - len(aggregated_closes)) / len(aggregated_closes)
            < 1e-2
        )

        for i in range(len(aggregated_closes)):
            start, end = aggregate_indices[i, :]

            np.testing.assert_allclose(closes[end - 1], aggregated_closes[i])
            np.testing.assert_allclose(np.min(lows[start:end]), aggregated_lows[i])
            np.testing.assert_allclose(np.max(highs[start:end]), aggregated_highs[i])
            np.testing.assert_allclose(times[end - 1] - times[start], 59.0 * 60.0)
            assert displacements[start] == displacement

        for i in range(len(split_indices) - 1):
            start = aggregate_indices[split_indices[i], 0]
            end = aggregate_indices[split_indices[i + 1] - 1, 1]

            time_diffs = np.diff(times[start:end])
            assert np.all(time_diffs <= 59.0 * 60.0)

            if split_indices[i + 1] < len(aggregated_closes):
                next_start = aggregate_indices[split_indices[i + 1], 0]
                assert next_start > end

    @given(
        fname=st.sampled_from(
            glob.glob(
                constants.DATA_STORAGE_LOCATION
                + "/"
                + constants.LOCAL_DATA_DIR
                + "/*/*.csv"
            )
        ),
        displacement=st.integers(0, 59),
    )
    @settings(max_examples=1, deadline=None)
    def test_against_naive(self, fname, displacement):
        fname_split = fname.split("/")
        symbol = fname_split[-1].split("_")[0]
        market = symbol + "/USD"
        coin = utils.get_coin(symbol)

        data_dir = fname.split(coin)[0]

        price_data = data.load_price_data(data_dir, market, return_price_data_only=True)

        closes = price_data["close"].values
        lows = price_data["low"].values
        highs = price_data["high"].values
        times = price_data["time"].values
        displacements = utils.get_displacements(price_data)

        return_tuple1 = utils.aggregate_from_displacement_naive(
            closes, lows, highs, times, displacements, displacement
        )
        return_tuple2 = utils.aggregate_from_displacement(
            closes, lows, highs, times, displacements, displacement
        )

        assert len(return_tuple1) == len(return_tuple2)

        for i in range(len(return_tuple1)):
            np.testing.assert_allclose(return_tuple1[i], return_tuple2[i])
