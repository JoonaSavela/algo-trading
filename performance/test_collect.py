import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algtra.collect import data
from algtra import utils
from algtra import constants
import time
import numpy as np


def test_calculate_max_average_spread_performance():
    distribution = np.random.normal(size=int(1e5))
    distribution = np.exp(distribution)
    total_balance = 2e3

    distributions1 = {
        "asks": distribution,
        "bids": distribution,
    }

    start = time.perf_counter()
    max_avg_spread1 = data.calculate_max_average_spread_naive(
        distributions1, total_balance
    )
    end = time.perf_counter()
    elapsed_naive = end - start

    distributions2 = np.stack([distribution, distribution], axis=1)

    start = time.perf_counter()
    max_avg_spread2 = data.calculate_max_average_spread(distributions2, total_balance)
    end = time.perf_counter()
    elapsed_compilation = end - start

    start = time.perf_counter()
    max_avg_spread2 = data.calculate_max_average_spread(distributions2, total_balance)
    end = time.perf_counter()
    elapsed = end - start

    utils.print_times(
        f"max_average_spread ({max_avg_spread1})",
        ["Benchmark", "Compilation", "Target"],
        [elapsed_naive, elapsed_compilation, elapsed],
    )

    np.testing.assert_almost_equal(max_avg_spread1, max_avg_spread2)
    assert elapsed <= elapsed_naive
