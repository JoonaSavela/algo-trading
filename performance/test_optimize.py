import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import algtra.optimize.utils as opt
from algtra import utils
from algtra import constants
import time
import numpy as np


def test_get_stop_loss_index_performance():
    lows = np.random.normal(size=int(1e4))
    lows = np.exp(lows)
    stop_loss = np.random.rand(1).item() / 20
    # stop_loss = 0.0

    start = time.perf_counter()
    stop_loss_index1 = opt.get_stop_loss_index_naive(lows, stop_loss)
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    stop_loss_index2 = opt.get_stop_loss_index(lows, stop_loss)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    stop_loss_index2 = opt.get_stop_loss_index(lows, stop_loss)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"get_stop_loss_index ({stop_loss_index1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    assert stop_loss_index1 == stop_loss_index2
    assert elapsed <= elapsed_naive


def test_get_stop_loss_indices_performance():
    N = int(1e4)

    closes = np.random.normal(size=N)
    closes = np.exp(closes)
    closes_to_lows = np.random.rand(N)
    lows = closes * closes_to_lows

    stop_loss = np.random.rand(1).item() / 20
    # stop_loss = 0.0
    aggregate_N = 12

    start = time.perf_counter()
    stop_loss_indices1 = opt.get_stop_loss_indices_naive(
        closes, lows, stop_loss, aggregate_N
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    stop_loss_indices2 = opt.get_stop_loss_indices(closes, lows, stop_loss, aggregate_N)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    stop_loss_indices2 = opt.get_stop_loss_indices(closes, lows, stop_loss, aggregate_N)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        "get_stop_loss_indices",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_array_equal(stop_loss_indices1, stop_loss_indices2)
    assert elapsed <= elapsed_naive


def test_get_take_profit_index_performance():
    highs = np.random.normal(size=int(1e4))
    highs = np.exp(highs)
    take_profit = np.random.rand(1).item() / 10
    take_profit = 1 / take_profit if take_profit > 0 else 1e7
    # take_profit = 1e7

    start = time.perf_counter()
    take_profit_index1 = opt.get_take_profit_index_naive(highs, take_profit)
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    take_profit_index2 = opt.get_take_profit_index(highs, take_profit)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    take_profit_index2 = opt.get_take_profit_index(highs, take_profit)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"get_take_profit_index ({take_profit_index1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    assert take_profit_index1 == take_profit_index2
    assert elapsed <= elapsed_naive


def test_get_take_profit_indices_performance():
    N = int(1e4)

    closes = np.random.normal(size=N)
    closes = np.exp(closes)
    closes_to_highs = np.random.rand(N)
    highs = closes * closes_to_highs

    take_profit = np.random.rand(1).item() / 10
    take_profit = 1 / take_profit if take_profit > 0 else 1e7
    # take_profit = 1e7
    aggregate_N = 12

    start = time.perf_counter()
    take_profit_indices1 = opt.get_take_profit_indices_naive(
        closes, highs, take_profit, aggregate_N
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    take_profit_indices2 = opt.get_take_profit_indices(
        closes, highs, take_profit, aggregate_N
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    take_profit_indices2 = opt.get_take_profit_indices(
        closes, highs, take_profit, aggregate_N
    )
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        "get_take_profit_indices",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_array_equal(take_profit_indices1, take_profit_indices2)
    assert elapsed <= elapsed_naive


def test_get_balanced_trade_performance():
    closes = np.random.lognormal(size=10000)

    stop_loss = np.random.rand(1).item()
    take_profit = np.random.rand(1).item()
    target_usd_percentage = np.random.rand(1).item()

    balancing_period = np.random.randint(1, 24, 1).item()

    distribution = np.random.normal(size=int(1e5))
    distribution = np.exp(distribution)
    orderbook_distributions = np.stack([distribution, distribution], axis=1)
    balance = 2e3

    taxes = bool(np.random.rand(1) < 0.5)
    tax_exemption = bool(np.random.rand(1) < 0.5)

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
        orderbook_distributions,
        balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )

    start = time.perf_counter()
    usd_values1, asset_values1, buy_sizes1 = opt.get_balanced_trade_naive(
        closes,
        stop_loss_index,
        take_profit_index,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        orderbook_distributions,
        balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    usd_values2, asset_values2, buy_sizes2 = opt.get_balanced_trade(
        closes,
        stop_loss_index,
        take_profit_index,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        orderbook_distributions,
        balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    usd_values2, asset_values2, buy_sizes2 = opt.get_balanced_trade(
        closes,
        stop_loss_index,
        take_profit_index,
        stop_loss,
        take_profit,
        target_usd_percentage,
        balancing_period,
        orderbook_distributions,
        balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        "get_take_profit_indices",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_allclose(usd_values1, usd_values2)
    np.testing.assert_allclose(asset_values1, asset_values2)
    np.testing.assert_almost_equal(buy_sizes1, buy_sizes2)
    assert elapsed <= elapsed_naive
