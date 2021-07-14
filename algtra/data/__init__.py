import pandas as pd
import numpy as np
from ftx.rest.client import FtxClient
import keys
from ciso8601 import parse_datetime


def get_coins():
    return ["BTC", "ETH"]


def datetime_to_timestamp(dt):
    return parse_datetime(dt).timestamp()


class PriceDataProcessor:
    def __init__(self) -> None:
        self.client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)

    def _get_price_data_and_subtract_its_length_from_limit(
        self, coin, limit, start_time=None
    ):
        market = coin + "/USD"

        X = self.client.get_historical_prices(
            market=market,
            resolution=60,
            limit=5000 if limit is None else min(limit, 5000),
            end_time=start_time,
        )
        if limit is not None:
            limit -= len(X)

        return X, limit

    def _get_start_time(self, X):
        start_time = min(parse_datetime(x["startTime"]) for x in X)
        start_time = start_time.timestamp()

        return start_time

    def fetch(self, coin: str, limit=None, stop_at_timestamp=None):
        if coin not in get_coins():
            raise ValueError(
                f"Invalid input for parameter 'coin' (value {coin} was given)."
            )

        if limit <= 0:
            raise ValueError(f"'limit' should be strictly positive.")

        price_data = []

        X, limit = self._get_price_data_and_subtract_its_length_from_limit(
            coin, limit, start_time=None
        )
        price_data.extend(X)

        if len(X) > 0:
            start_time = self._get_start_time(X)

            while (
                len(X) > 1
                and (limit is None or limit > 0)
                and (stop_at_timestamp is None or stop_at_timestamp < start_time)
            ):
                X, limit = self._get_price_data_and_subtract_its_length_from_limit(
                    coin, limit, start_time=start_time
                )
                price_data.extend(X)

                start_time = self._get_start_time(X)

            price_data = (
                pd.DataFrame(price_data)
                .drop_duplicates("startTime")
                .sort_values("startTime")
            )
            price_data["time"] = price_data["startTime"].map(datetime_to_timestamp)
            price_data = price_data.drop("startTime", axis=1)

            if stop_at_timestamp is not None:
                price_data = price_data[price_data["time"] > stop_at_timestamp]

        else:
            price_data = pd.DataFrame(
                price_data,
                columns=(
                    "close",
                    "open",
                    "high",
                    "low",
                    "volume",
                    "time",
                ),
            )

        return price_data
