import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from hypothesis import given, assume, example, strategies as st, settings
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype
import algtra.optimize.utils as opt
from algtra import constants
from algtra.collect import data
from datetime import datetime, timedelta
import json
import pytest


@pytest.mark.slow
@given(k=st.integers(1, 1000), min_length=st.integers(0, constants.MINUTES_IN_A_YEAR))
@settings(deadline=None)
def test_get_symbols_for_optimization(k, min_length):
    symbols = opt.get_symbols_for_optimization()

    k = min(k, len(symbols))

    symbols_with_k = opt.get_symbols_for_optimization(k=k)
    symbols_with_k_and_min_length = opt.get_symbols_for_optimization(
        k=k, min_length=min_length
    )

    assert len(symbols_with_k) == k
    assert len(symbols_with_k_and_min_length) <= k
    assert len(symbols_with_k) <= len(symbols)
    assert len(symbols_with_k_and_min_length) <= len(symbols)

    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    volumes_fname = os.path.abspath(os.path.join(data_dir, "volumes.json"))

    with open(volumes_fname, "r") as file:
        volumes = json.load(file)

    volumes_list = [volumes[symbol] for symbol in symbols_with_k_and_min_length]

    assert np.all(np.diff(volumes_list) <= 0)

    for symbol in symbols_with_k_and_min_length:
        data_length = data.load_price_data(
            data_dir, symbol + "/USD", return_price_data=False
        )[1]

        assert data_length >= min_length


class Test_quantile_objective_function:
    @given(
        log_profits=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(-1e1, 1e1, width=64),
        ),
        shift=st.floats(0.0, 10.0),
        quantile=st.floats(0.0, 1.0),
    )
    def test_distribution_shift(self, log_profits, shift, quantile):
        assume(shift > 0.0)
        assert np.quantile(log_profits + shift, quantile) > np.quantile(
            log_profits, quantile
        )

    @given(
        log_profits=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(-1e1, 1e1, width=64),
        ),
        scale=st.floats(0.01, 1.0),
        quantile=st.floats(0.0, 1.0),
    )
    def test_distribution_scale(self, log_profits, scale, quantile):
        log_profits_median = np.median(log_profits)

        scaled_log_profits = log_profits - log_profits_median
        scaled_log_profits *= scale
        scaled_log_profits += log_profits_median

        scaled_q = np.quantile(scaled_log_profits, quantile)
        q = np.quantile(log_profits, quantile)

        atol = 1e-7
        if quantile <= 0.5:
            assert scaled_q >= q or np.abs(scaled_q - q) < atol
        else:
            assert scaled_q <= q or np.abs(scaled_q - q) < atol


@given(
    lows=arrays(
        np.float64,
        st.integers(0, 100),
        elements=st.floats(0.0, 1e7, width=64),
    ),
    stop_loss=st.floats(0.0, 1.0),
)
@settings(deadline=timedelta(milliseconds=1000))
def test_get_stop_loss_index_against_naive(lows, stop_loss):
    assert opt.get_stop_loss_index_naive(lows, stop_loss) == opt.get_stop_loss_index(
        lows, stop_loss
    )


@pytest.mark.slow
@given(
    N=st.integers(0, 1000),
    aggregate_N=st.integers(2, 12),
    stop_loss=st.floats(0.0, 1.0),
    probe_N=st.integers(0, 1000),
)
def test_get_stop_loss_indices_naive(N, aggregate_N, stop_loss, probe_N):
    assume(stop_loss < 1.0)

    closes = np.ones(N)
    lows = np.ones(N)
    idx = np.arange(min(aggregate_N, N))

    stop_loss_indices = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )

    np.testing.assert_array_equal(
        stop_loss_indices, np.ones(len(stop_loss_indices)) * N - idx
    )

    if N > 0:
        assume(probe_N < N)
        lows[probe_N] = 0.0

        stop_loss_indices = opt.get_stop_loss_indices_naive(
            closes, lows, stop_loss, aggregate_N
        )

        li = idx <= probe_N

        np.testing.assert_array_equal(
            stop_loss_indices[li], np.ones(np.sum(li)) * probe_N - idx[li]
        )
        np.testing.assert_array_equal(
            stop_loss_indices[~li], np.ones(np.sum(~li)) * N - idx[~li]
        )

    lows = np.zeros(N)

    stop_loss_indices = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )

    np.testing.assert_array_equal(stop_loss_indices, np.zeros(len(stop_loss_indices)))


@pytest.mark.slow
@given(
    closes=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(0.001, 1e3, width=64),
    ),
    closes_to_lows=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(0.0, 1.0, width=64),
    ),
    stop_loss=st.floats(0.0, 1.0),
    aggregate_N=st.integers(2, 12),
)
@settings(deadline=timedelta(milliseconds=1000))
def test_get_stop_loss_indices_against_naive(
    closes, closes_to_lows, stop_loss, aggregate_N
):
    assume(len(closes) <= len(closes_to_lows))
    closes_to_lows = closes_to_lows[: len(closes)]
    lows = closes * closes_to_lows

    stop_loss_indices1 = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )
    stop_loss_indices2 = opt.get_stop_loss_indices(closes, lows, stop_loss, aggregate_N)

    np.testing.assert_array_equal(stop_loss_indices1, stop_loss_indices2)


@given(
    highs=arrays(
        np.float64,
        st.integers(0, 100),
        elements=st.floats(0.0, 1e7, width=64),
    ),
    take_profit=st.floats(1.0),
)
@settings(deadline=timedelta(milliseconds=1000))
def test_get_take_profit_index_against_naive(highs, take_profit):
    assert opt.get_take_profit_index_naive(
        highs, take_profit
    ) == opt.get_take_profit_index(highs, take_profit)


@pytest.mark.slow
@given(
    N=st.integers(0, 1000),
    aggregate_N=st.integers(2, 12),
    take_profit=st.floats(1.0),
    probe_N=st.integers(0, 1000),
)
def test_get_take_profit_indices_naive(N, aggregate_N, take_profit, probe_N):
    assume(take_profit > 1.0)

    closes = np.ones(N)
    highs = np.ones(N)
    idx = np.arange(min(aggregate_N, N))

    take_profit_indices = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )

    np.testing.assert_array_equal(
        take_profit_indices, np.ones(len(take_profit_indices)) * N - idx
    )

    if N > 0:
        assume(probe_N < N)
        highs[probe_N] = np.Inf

        take_profit_indices = opt.get_take_profit_indices_naive(
            closes, highs, take_profit, aggregate_N
        )

        li = idx <= probe_N

        np.testing.assert_array_equal(
            take_profit_indices[li], np.ones(np.sum(li)) * probe_N - idx[li]
        )
        np.testing.assert_array_equal(
            take_profit_indices[~li], np.ones(np.sum(~li)) * N - idx[~li]
        )

    highs = np.ones(N) * np.Inf

    take_profit_indices = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )

    np.testing.assert_array_equal(
        take_profit_indices, np.zeros(len(take_profit_indices))
    )


@pytest.mark.slow
@given(
    closes=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(0.001, 1e3, width=64),
    ),
    closes_to_highs=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(1.0, 100.0, width=64),
    ),
    take_profit=st.floats(1.0),
    aggregate_N=st.integers(2, 12),
)
@settings(deadline=timedelta(milliseconds=1000))
def test_get_take_profit_indices_against_naive(
    closes, closes_to_highs, take_profit, aggregate_N
):
    assume(len(closes) <= len(closes_to_highs))
    closes_to_highs = closes_to_highs[: len(closes)]
    highs = closes * closes_to_highs

    take_profit_indices1 = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )
    take_profit_indices2 = opt.get_take_profit_indices(
        closes, highs, take_profit, aggregate_N
    )

    np.testing.assert_array_equal(take_profit_indices1, take_profit_indices2)


class Test_get_balanced_trade:
    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 1000),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        target_usd_percentage=st.floats(0.0, 1.0),
    )
    @settings(deadline=None)
    def test_naive_base_case(
        self,
        closes,
        target_usd_percentage,
    ):
        (usd_values, asset_values, _,) = opt.get_balanced_trade_naive(
            closes,
            len(closes),
            len(closes),
            0.0,
            100.0,
            target_usd_percentage,
            balancing_period=len(closes) + 2,
            orderbook_distributions=np.array([[0.0, 0.0]]),
            balance=0.0,
            taxes=False,
        )

        trade = usd_values + asset_values

        for i in range(len(closes)):
            np.testing.assert_allclose(
                trade[i],
                closes[i] * (1.0 - target_usd_percentage) + target_usd_percentage,
            )

    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 1000),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        stop_loss=st.floats(0.0, 0.99),
        take_profit=st.floats(1.01, 1e6),
        target_usd_percentage=st.floats(0.0, 1.0),
    )
    def test_naive_stop_loss_and_take_profit(
        self,
        closes,
        stop_loss,
        take_profit,
        target_usd_percentage,
    ):
        stop_loss_index = opt.get_stop_loss_index(closes, stop_loss)
        take_profit_index = opt.get_take_profit_index(closes, take_profit)

        usd_values, asset_values, _ = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period=len(closes) + 2,
            orderbook_distributions=np.array([[0.0, 0.0]]),
            balance=0.0,
            taxes=False,
        )

        trigger_i = min(stop_loss_index, take_profit_index)
        triggered = trigger_i < len(closes)

        trade = usd_values + asset_values

        if triggered:
            stop_loss_is_before_take_profit = stop_loss_index <= take_profit_index
            trigger_value = (
                stop_loss if stop_loss_is_before_take_profit else take_profit
            )

            np.testing.assert_allclose(
                trade[trigger_i:],
                trigger_value * (1.0 - target_usd_percentage) + target_usd_percentage,
            )

    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        target_usd_percentage=st.floats(0.0, 1.0),
        balancing_period=st.integers(1, 24),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        balance=st.floats(0.0, 1e6),
        taxes=st.booleans(),
        tax_exemption=st.booleans(),
        reverse=st.booleans(),
    )
    @settings(deadline=timedelta(seconds=5))
    def test_naive_balancing(
        self,
        closes,
        target_usd_percentage,
        balancing_period,
        distribution,
        balance,
        taxes,
        tax_exemption,
        reverse,
    ):
        assume(np.sum(distribution) > 0)
        distributions = np.stack([distribution, distribution], axis=1)

        closes = np.concatenate([np.ones(1), closes])
        log_closes = np.log(closes)
        log_closes = log_closes - (log_closes[-1] - log_closes[0]) * np.linspace(
            0.0, 1.0, len(closes)
        )
        np.testing.assert_allclose(log_closes[0], log_closes[-1])
        closes = np.exp(log_closes)
        closes = closes[1:]

        (
            usd_values_without_balancing,
            asset_values_without_balancing,
            _,
        ) = opt.get_balanced_trade_naive(
            closes,
            len(closes),
            len(closes),
            0.0,
            100.0,
            target_usd_percentage,
            balancing_period=len(closes) + 2,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        trade_without_balancing = (
            usd_values_without_balancing + asset_values_without_balancing
        )

        usd_values, asset_values, _ = opt.get_balanced_trade_naive(
            closes,
            len(closes),
            len(closes),
            0.0,
            100.0,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        trade = usd_values + asset_values

        rtol = 1e-7
        # should be trade[-1] >= trade_without_balancing[-1]
        if trade[-1] < trade_without_balancing[-1]:
            np.testing.assert_allclose(
                trade[-1], trade_without_balancing[-1], rtol=rtol
            )

        closes = np.concatenate([np.ones(1), closes])
        closes = np.sort(closes)
        if reverse:
            closes = np.flip(closes)
        closes /= closes[0]
        closes = closes[1:]

        (
            usd_values_without_balancing,
            asset_values_without_balancing,
            _,
        ) = opt.get_balanced_trade_naive(
            closes,
            len(closes),
            len(closes),
            0.0,
            100.0,
            target_usd_percentage,
            balancing_period=len(closes) + 2,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        trade_without_balancing = (
            usd_values_without_balancing + asset_values_without_balancing
        )

        usd_values, asset_values, _ = opt.get_balanced_trade_naive(
            closes,
            len(closes),
            len(closes),
            0.0,
            100.0,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=taxes,
            tax_exemption=tax_exemption,
        )

        trade = usd_values + asset_values

        # should be trade[-1] <= trade_without_balancing[-1]
        if trade[-1] > trade_without_balancing[-1]:
            np.testing.assert_allclose(
                trade[-1], trade_without_balancing[-1], rtol=rtol
            )

    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 1000),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        stop_loss=st.floats(0.0, 0.99),
        take_profit=st.floats(1.01, 1e6),
        target_usd_percentage=st.floats(0.0, 1.0),
        balancing_period=st.integers(1, 24),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0, 1e7, width=64),
        ),
        balance=st.floats(0.0, 1e6),
    )
    @settings(deadline=None)
    def test_naive_spread(
        self,
        closes,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        distribution,
        balance,
    ):
        assume(np.sum(distribution) > 0)
        distributions = np.stack([distribution, distribution], axis=1)

        stop_loss_index = opt.get_stop_loss_index(closes, stop_loss)
        take_profit_index = opt.get_take_profit_index(closes, take_profit)

        (
            usd_values_without_spread,
            asset_values_without_spread,
            _,
        ) = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=0.0,
            taxes=False,
        )

        trade_without_spread = usd_values_without_spread + asset_values_without_spread

        usd_values, asset_values, _ = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=False,
        )

        trade = usd_values + asset_values

        assert np.all(trade <= trade_without_spread)

    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        stop_loss=st.floats(0.0, 0.99),
        take_profit=st.floats(1.01, 1e3),
        target_usd_percentage=st.floats(0.0, 1.0),
        balancing_period=st.integers(1, 24),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e7, width=64),
        ),
        balance=st.floats(0.0, 1e6),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=None)
    def test_naive_taxes(
        self,
        closes,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        distribution,
        balance,
        tax_exemption,
    ):
        distributions = np.stack([distribution, distribution], axis=1)

        stop_loss_index = opt.get_stop_loss_index(closes, stop_loss)
        take_profit_index = opt.get_take_profit_index(closes, take_profit)

        (
            usd_values_without_taxes,
            asset_values_without_taxes,
            _,
        ) = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=False,
        )

        trade_without_taxes = usd_values_without_taxes + asset_values_without_taxes

        usd_values, asset_values, buy_sizes = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=True,
            tax_exemption=tax_exemption,
        )

        trade = usd_values + asset_values

        assert np.all(buy_sizes >= 0.0)
        assert np.all(buy_sizes <= 1e-7)

        if not tax_exemption:
            assert np.all(trade <= trade_without_taxes)

    @pytest.mark.slow
    @given(
        closes=arrays(
            np.float64,
            st.integers(1, 1000),
            elements=st.floats(0.001, 1e3, width=64),
        ),
        stop_loss=st.floats(0.0, 0.99),
        take_profit=st.floats(1.01, 1e3),
        target_usd_percentage=st.floats(0.0, 1.0),
        balancing_period=st.integers(1, 24),
        distribution=arrays(
            np.float64,
            st.integers(1, 100),
            elements=st.floats(0.001, 1e7, width=64),
        ),
        balance=st.floats(0.0, 1e6),
        tax_exemption=st.booleans(),
    )
    @settings(deadline=None)
    def test_against_naive(
        self,
        closes,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        distribution,
        balance,
        tax_exemption,
    ):
        distributions = np.stack([distribution, distribution], axis=1)

        stop_loss_index = opt.get_stop_loss_index(closes, stop_loss)
        take_profit_index = opt.get_take_profit_index(closes, take_profit)

        usd_values1, asset_values1, buy_sizes1 = opt.get_balanced_trade_naive(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=True,
            tax_exemption=tax_exemption,
        )

        usd_values2, asset_values2, buy_sizes2 = opt.get_balanced_trade(
            closes,
            stop_loss_index,
            take_profit_index,
            stop_loss,
            take_profit,
            target_usd_percentage,
            balancing_period,
            orderbook_distributions=distributions,
            balance=balance,
            taxes=True,
            tax_exemption=tax_exemption,
        )

        np.testing.assert_allclose(usd_values1, usd_values2)
        np.testing.assert_allclose(asset_values1, asset_values2)
        np.testing.assert_almost_equal(buy_sizes1, buy_sizes2)
