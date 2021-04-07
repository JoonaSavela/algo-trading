import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from hypothesis import given, assume, example, strategies as st, settings
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype
import algtra.optimize.utils as opt
from datetime import datetime, timedelta


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


@given(
    N=st.integers(0, 10000),
    aggregate_N=st.integers(2, 12),
    stop_loss=st.floats(0.0, 1.0),
    probe_N=st.integers(0, 10000),
)
def test_get_stop_loss_indices_naive(N, aggregate_N, stop_loss, probe_N):
    assume(stop_loss < 1.0)

    closes = np.ones(N)
    lows = np.ones(N)

    stop_loss_indices = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )

    np.testing.assert_array_equal(
        stop_loss_indices, np.ones(len(stop_loss_indices)) * N
    )

    if N > 0:
        assume(probe_N < N)
        lows[probe_N] = 0.0

        stop_loss_indices = opt.get_stop_loss_indices_naive(
            closes, lows, stop_loss, aggregate_N
        )

        idx = np.arange(len(stop_loss_indices))
        li = idx <= probe_N

        np.testing.assert_array_equal(
            stop_loss_indices[li], np.ones(np.sum(li)) * probe_N
        )
        np.testing.assert_array_equal(stop_loss_indices[~li], np.ones(np.sum(~li)) * N)

    lows = np.zeros(N)

    stop_loss_indices = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )

    np.testing.assert_array_equal(stop_loss_indices, np.arange(len(stop_loss_indices)))


@given(
    closes=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(0.0, 1e7, width=64),
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


@given(
    N=st.integers(0, 10000),
    aggregate_N=st.integers(2, 12),
    take_profit=st.floats(1.0),
    probe_N=st.integers(0, 10000),
)
def test_get_take_profit_indices_naive(N, aggregate_N, take_profit, probe_N):
    assume(take_profit > 1.0)

    closes = np.ones(N)
    highs = np.ones(N)

    take_profit_indices = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )

    np.testing.assert_array_equal(
        take_profit_indices, np.ones(len(take_profit_indices)) * N
    )

    if N > 0:
        assume(probe_N < N)
        highs[probe_N] = np.Inf

        take_profit_indices = opt.get_take_profit_indices_naive(
            closes, highs, take_profit, aggregate_N
        )

        idx = np.arange(len(take_profit_indices))
        li = idx <= probe_N

        np.testing.assert_array_equal(
            take_profit_indices[li], np.ones(np.sum(li)) * probe_N
        )
        np.testing.assert_array_equal(
            take_profit_indices[~li], np.ones(np.sum(~li)) * N
        )

    highs = np.ones(N) * np.Inf

    take_profit_indices = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )

    np.testing.assert_array_equal(
        take_profit_indices, np.arange(len(take_profit_indices))
    )


@given(
    closes=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(0.0, 1e7, width=64),
    ),
    closes_to_highs=arrays(
        np.float64,
        st.integers(0, 1000),
        elements=st.floats(1.0, 10000.0, width=64),
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
