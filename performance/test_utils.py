import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algtra import utils
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


def test_get_balanced_usd_and_asset_values_performance():
    usd_value = np.random.rand(1).item()
    asset_value = np.random.rand(1).item()
    target_usd_percentage = np.random.rand(1).item()

    distribution = np.random.normal(size=int(1e5))
    distribution = np.exp(distribution)
    distributions = np.stack([distribution, distribution], axis=1)
    balance = 2e3

    taxes = bool(np.random.rand(1) < 0.5)
    tax_exemption = bool(np.random.rand(1) < 0.5)

    start = time.perf_counter()
    usd_value1, asset_value1 = utils.get_balanced_usd_and_asset_values_naive(
        usd_value,
        asset_value,
        target_usd_percentage,
        orderbook_distributions=distributions,
        balance=balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    start = time.perf_counter()
    usd_value2, asset_value2 = utils.get_balanced_usd_and_asset_values(
        usd_value,
        asset_value,
        target_usd_percentage,
        orderbook_distributions=distributions,
        balance=balance,
        taxes=taxes,
        tax_exemption=tax_exemption,
    )
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    usd_value2, asset_value2 = utils.get_balanced_usd_and_asset_values(
        usd_value,
        asset_value,
        target_usd_percentage,
        orderbook_distributions=distributions,
        balance=balance,
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
    assert elapsed <= elapsed_naive
