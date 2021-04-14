import numpy as np
import keys
from ftx.rest.client import FtxClient
from algtra.collect import data
import os
from algtra import constants, utils
import algtra.optimize.utils as opt
import matplotlib.pyplot as plt
from ciso8601 import parse_datetime
import time


def main():
    data_dir = os.path.abspath(
        os.path.join(constants.DATA_STORAGE_LOCATION, constants.LOCAL_DATA_DIR)
    )

    price_data = data.load_price_data(data_dir, "BTC/USD", return_price_data_only=True)
    print(len(price_data))
    print()

    closes = price_data["close"].values
    lows = price_data["low"].values
    highs = price_data["high"].values
    times = price_data["time"].values
    displacements = utils.get_displacements(price_data)
    displacement = 55

    start = time.perf_counter()

    return_tuple = utils.aggregate_from_displacement(
        closes, lows, highs, times, displacements, displacement
    )
    end = time.perf_counter()

    print(f"Elapsed: {end - start} (s)")
    print()

    for result in return_tuple:
        print(result)
        print(len(result))
        print()

    aggregated_closes = return_tuple[0]
    print(abs(len(closes) // 60 - len(aggregated_closes)) / len(aggregated_closes))
    print()


if __name__ == "__main__":
    main()
