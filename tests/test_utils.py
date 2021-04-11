import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from algtra import utils
from algtra.collect import data
from algtra.constants import PARTIAL_LEVERAGED_SYMBOLS
from ftx.rest.client import FtxClient
import keys
from hypothesis import given, assume, example, strategies as st, settings
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype
from datetime import timedelta
import pytest


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


class Test_apply_taxes_to_value_change:
    @given(
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        target_usd_percentage=st.floats(0, 1.0),
        sell_price=st.floats(1e-3, 1e3),
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
    )
    def test_naive(
        self,
        usd_value,
        asset_value,
        target_usd_percentage,
        sell_price,
        buy_prices,
        buy_sizes,
    ):
        if len(buy_sizes) <= len(buy_prices):
            buy_prices = buy_prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(buy_prices)]
        assume(usd_value + asset_value > 1e-3)

        total_value = usd_value + asset_value
        new_usd_percentage = usd_value / total_value

        percentage_change = target_usd_percentage - new_usd_percentage
        value_change = percentage_change * total_value

        assume(value_change > 0.0)
        assume(np.dot(buy_sizes, buy_prices) >= total_value)

        value_change, _ = utils.apply_taxes_to_value_change_naive(
            value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
        )

        assert 0 <= value_change <= asset_value

    @given(
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        target_usd_percentage=st.floats(0, 1.0),
        sell_price=st.floats(1e-3, 1e3),
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
    )
    @settings(deadline=timedelta(milliseconds=500))
    def test_against_naive(
        self,
        usd_value,
        asset_value,
        target_usd_percentage,
        sell_price,
        buy_prices,
        buy_sizes,
    ):
        if len(buy_sizes) <= len(buy_prices):
            buy_prices = buy_prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(buy_prices)]
        assume(usd_value + asset_value > 1e-3)

        total_value = usd_value + asset_value
        new_usd_percentage = usd_value / total_value

        percentage_change = target_usd_percentage - new_usd_percentage
        value_change = percentage_change * total_value

        assume(value_change > 0.0)
        assume(np.dot(buy_sizes, buy_prices) > total_value)

        value_change1, _ = utils.apply_taxes_to_value_change_naive(
            value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
        )

        value_change2, _ = utils.apply_taxes_to_value_change(
            value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
        )

        np.testing.assert_allclose(value_change1, value_change2)


class Test_get_balanced_usd_and_asset_values:
    @given(
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        target_usd_percentage=st.floats(0, 1.0),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e7, width=64),
        ),
        balance=st.floats(0, 1e6),
        prices=arrays(
            np.float64,
            st.integers(1, 10),
            elements=st.floats(1e-3, 1e3, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 10),
            elements=st.floats(1e-3, 1e3, width=64),
        ),
        buy_sizes_index=st.integers(1, 9),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=timedelta(milliseconds=2000))
    def test_naive(
        self,
        usd_value,
        asset_value,
        target_usd_percentage,
        distribution,
        balance,
        prices,
        buy_sizes,
        buy_sizes_index,
        tax_exemption,
    ):
        if len(buy_sizes) <= len(prices):
            prices = prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(prices)]
        buy_sizes_index = min(buy_sizes_index, len(prices) - 1)

        total_value = usd_value + asset_value
        assume(total_value > 1e-3)
        assume(
            np.dot(buy_sizes[:buy_sizes_index], prices[:buy_sizes_index]) >= total_value
        )

        new_usd_percentage = usd_value / total_value

        percentage_change = target_usd_percentage - new_usd_percentage
        value_change = percentage_change * total_value

        distributions = np.stack([distribution, distribution], axis=1)

        (
            usd_value_base_case,
            asset_value_base_case,
            new_buy_size,
            buy_size_diffs,
        ) = utils.get_balanced_usd_and_asset_values_naive(
            usd_value,
            asset_value,
            target_usd_percentage,
            orderbook_distributions=None,
            balance=0.0,
            prices=prices,
            buy_sizes=buy_sizes,
            buy_sizes_index=buy_sizes_index,
            taxes=False,
        )

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
        ) = utils.get_balanced_usd_and_asset_values_naive(
            usd_value,
            asset_value,
            target_usd_percentage,
            orderbook_distributions=distributions,
            balance=balance,
            prices=prices,
            buy_sizes=buy_sizes,
            buy_sizes_index=buy_sizes_index,
            taxes=False,
        )

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
        ) = utils.get_balanced_usd_and_asset_values_naive(
            usd_value,
            asset_value,
            target_usd_percentage,
            orderbook_distributions=distributions,
            balance=balance,
            prices=prices,
            buy_sizes=buy_sizes,
            buy_sizes_index=buy_sizes_index,
            taxes=True,
            tax_exemption=tax_exemption,
        )

        np.testing.assert_almost_equal(
            usd_value_full_case / (usd_value_full_case + asset_value_full_case),
            target_usd_percentage,
        )
        if not tax_exemption:
            assert (
                usd_value_full_case + asset_value_full_case
                <= usd_value_with_spread + asset_value_with_spread
            )

    @given(
        usd_value=st.floats(0, 1.0),
        asset_value=st.floats(0, 1.0),
        target_usd_percentage=st.floats(0, 1.0),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e7, width=64),
        ),
        balance=st.floats(0, 1e6),
        prices=arrays(
            np.float64,
            st.integers(1, 10),
            elements=st.floats(1e-3, 1e3, width=64),
        ),
        buy_sizes=arrays(
            np.float64,
            st.integers(1, 10),
            elements=st.floats(1e-3, 1e3, width=64),
        ),
        buy_sizes_index=st.integers(1, 9),
        taxes=st.booleans(),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=timedelta(milliseconds=2000))
    def test_against_naive(
        self,
        usd_value,
        asset_value,
        target_usd_percentage,
        distribution,
        balance,
        prices,
        buy_sizes,
        buy_sizes_index,
        taxes,
        tax_exemption,
    ):
        if len(buy_sizes) <= len(prices):
            prices = prices[: len(buy_sizes)]
        else:
            buy_sizes = buy_sizes[: len(prices)]
        buy_sizes_index = min(buy_sizes_index, len(prices) - 1)

        total_value = usd_value + asset_value
        assume(total_value > 1e-3)
        assume(
            np.dot(buy_sizes[:buy_sizes_index], prices[:buy_sizes_index]) >= total_value
        )

        new_usd_percentage = usd_value / total_value

        percentage_change = target_usd_percentage - new_usd_percentage
        value_change = percentage_change * total_value

        distributions = np.stack([distribution, distribution], axis=1)

        (
            usd_value1,
            asset_value1,
            new_buy_size1,
            buy_size_diffs1,
        ) = utils.get_balanced_usd_and_asset_values_naive(
            usd_value,
            asset_value,
            target_usd_percentage,
            orderbook_distributions=distributions,
            balance=balance,
            prices=prices,
            buy_sizes=buy_sizes,
            buy_sizes_index=buy_sizes_index,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        (
            usd_value2,
            asset_value2,
            new_buy_size2,
            buy_size_diffs2,
        ) = utils.get_balanced_usd_and_asset_values(
            usd_value,
            asset_value,
            target_usd_percentage,
            orderbook_distributions=distributions,
            balance=balance,
            prices=prices,
            buy_sizes=buy_sizes,
            buy_sizes_index=buy_sizes_index,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        assert usd_value1 == usd_value2
        assert asset_value1 == asset_value2
        assert new_buy_size1 == new_buy_size2
        np.testing.assert_allclose(buy_size_diffs1, buy_size_diffs2)


@given(logit=st.floats(-20, 20))
def test_roundtrip_logit_to_p_p_to_logit(logit):
    value0 = utils.logit_to_p(logit=logit)
    value1 = utils.p_to_logit(p=value0)
    np.testing.assert_allclose(logit, value1)


class Test_get_symbol_and_get_coin:
    def test_get_coin(self):
        client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
        markets = data.get_filtered_markets(client)

        for curr in markets["baseCurrency"]:
            coin = utils.get_coin(curr)
            for partial_leveraged_symbol in PARTIAL_LEVERAGED_SYMBOLS:
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
