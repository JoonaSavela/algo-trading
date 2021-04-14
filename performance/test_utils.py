import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algtra import utils
from algtra.collect import data
from algtra import constants
import time
import numpy as np


def test_calculate_max_average_spread_performance():
    distribution = np.random.normal(size=int(1e5))
    distribution = np.exp(distribution)
    balance = 2e3

    distributions1 = {
        "asks": distribution,
        "bids": distribution,
    }

    start = time.perf_counter()
    max_avg_spread1 = utils.calculate_max_average_spread_naive(distributions1, balance)
    end = time.perf_counter()
    elapsed_naive = end - start

    distributions2 = np.stack([distribution, distribution], axis=1)

    start = time.perf_counter()
    max_avg_spread2 = utils.calculate_max_average_spread(distributions2, balance)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    max_avg_spread2 = utils.calculate_max_average_spread(distributions2, balance)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"max_average_spread ({max_avg_spread1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_almost_equal(max_avg_spread1, max_avg_spread2)
    assert elapsed <= elapsed_naive


def test_get_average_buy_price_performance():
    N = 1000
    buy_prices = np.random.normal(size=N)
    buy_prices = np.exp(buy_prices)
    buy_sizes = np.random.normal(size=N)
    buy_sizes = np.exp(buy_sizes)
    sell_size = min(1e3, np.sum(buy_sizes))

    start = time.perf_counter()
    avg_buy_price1, buy_size_diffs1 = utils.get_average_buy_price_naive(
        buy_prices, buy_sizes, sell_size
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    avg_buy_price2, buy_size_diffs2 = utils.get_average_buy_price(
        buy_prices, buy_sizes, sell_size
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    avg_buy_price2, buy_size_diffs2 = utils.get_average_buy_price(
        buy_prices, buy_sizes, sell_size
    )
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"get_average_buy_price ({avg_buy_price1}, {sell_size})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_almost_equal(avg_buy_price1, avg_buy_price2)
    np.testing.assert_almost_equal(buy_size_diffs1, buy_size_diffs2)
    assert elapsed <= elapsed_naive


def test_apply_taxes_to_value_change_performance():
    N = 1000
    buy_prices = np.random.lognormal(size=N)
    buy_sizes = np.random.lognormal(size=N)
    target_usd_percentage = np.random.rand(1).item()
    sell_price = np.random.rand(1).item()

    usd_value = np.random.rand(1).item()
    asset_value = np.random.rand(1).item()

    total_value = usd_value + asset_value
    new_usd_percentage = usd_value / total_value

    percentage_change = target_usd_percentage - new_usd_percentage
    value_change = percentage_change * total_value

    while value_change <= 0.0:
        usd_value = np.random.rand(1).item()
        asset_value = np.random.rand(1).item()

        total_value = usd_value + asset_value
        new_usd_percentage = usd_value / total_value

        percentage_change = target_usd_percentage - new_usd_percentage
        value_change = percentage_change * total_value

    start = time.perf_counter()
    value_change1, _ = utils.apply_taxes_to_value_change_naive(
        value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    value_change2, _ = utils.apply_taxes_to_value_change(
        value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    value_change2, _ = utils.apply_taxes_to_value_change(
        value_change, sell_price, buy_prices, buy_sizes, target_usd_percentage
    )
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"apply_taxes_to_value_change ({value_change1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_almost_equal(value_change1, value_change2)
    assert elapsed <= elapsed_naive


def test_get_balanced_usd_and_asset_values_performance():
    usd_value = np.random.rand(1).item()
    asset_value = np.random.rand(1).item()
    target_usd_percentage = np.random.rand(1).item()

    distribution = np.random.normal(size=int(1e5))
    distribution = np.exp(distribution)
    distributions = np.stack([distribution, distribution], axis=1)
    balance = 2e3

    price = np.random.rand(1).item()
    prices = np.array([price, price])

    total_value = usd_value + asset_value
    new_usd_percentage = usd_value / total_value

    percentage_change = target_usd_percentage - new_usd_percentage
    value_change = percentage_change * total_value

    buy_size = max(np.random.rand(1).item(), value_change / price)
    buy_sizes = np.array([buy_size])
    buy_sizes_index = 1

    taxes = bool(np.random.rand(1) < 0.5)
    tax_exemption = bool(np.random.rand(1) < 0.5)

    utils.get_balanced_usd_and_asset_values_naive(
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

    start = time.perf_counter()
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
    end = time.perf_counter()
    elapsed_naive = end - start

    buy_sizes = np.array([buy_size])

    start = time.perf_counter()
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
    end = time.perf_counter()
    elapsed_compilation = end - start

    buy_sizes = np.array([buy_size])

    start = time.perf_counter()
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
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"get_balanced_usd_and_asset_values ({usd_value1}, {asset_value1}, {target_usd_percentage})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    assert usd_value1 == usd_value2
    assert asset_value1 == asset_value2
    assert new_buy_size1 == new_buy_size2
    np.testing.assert_allclose(buy_size_diffs1, buy_size_diffs2)
    assert elapsed <= elapsed_naive


def test_get_next_displacement_index_performance():
    N = 10000
    displacements = np.random.randint(0, 1000000, size=N)
    start_i = np.random.randint(0, N)
    displacement = np.random.randint(0, 1000000)

    start = time.perf_counter()
    i1 = utils.get_next_displacement_index_naive(displacements, start_i, displacement)
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    i2 = utils.get_next_displacement_index(displacements, start_i, displacement)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    i2 = utils.get_next_displacement_index(displacements, start_i, displacement)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"get_next_displacement_index ({i1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    assert i1 == i2
    assert elapsed <= elapsed_naive


def test_aggregate_from_displacement_performance():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    price_data = data.load_price_data(data_dir, "BTC/USD", return_price_data_only=True)

    closes = price_data["close"].values
    lows = price_data["low"].values
    highs = price_data["high"].values
    times = price_data["time"].values
    displacements = utils.get_displacements(price_data)
    displacement = 55

    start = time.perf_counter()
    return_tuple1 = utils.aggregate_from_displacement_naive(
        closes, lows, highs, times, displacements, displacement
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    return_tuple2 = utils.aggregate_from_displacement(
        closes, lows, highs, times, displacements, displacement
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    return_tuple2 = utils.aggregate_from_displacement(
        closes, lows, highs, times, displacements, displacement
    )
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"aggregate_from_displacement",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    assert len(return_tuple1) == len(return_tuple2)

    for i in range(len(return_tuple1)):
        np.testing.assert_allclose(return_tuple1[i], return_tuple2[i])
    assert elapsed <= elapsed_naive
