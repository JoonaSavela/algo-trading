from algtra import constants
import pandas as pd


def test_get_filtered_markets_should_return_pd_DataFrame(
    dp,
):
    assert isinstance(dp.get_filtered_markets(), pd.DataFrame)


def test_get_filtered_markets_should_have_USD_as_quote_currency(dp):
    markets = dp.get_filtered_markets()

    assert all(markets["quoteCurrency"] == "USD")
    assert all(markets["name"].str.contains("/USD"))


def test_get_filtered_markets_names_should_not_have_USDT_as_quote_currency(dp):
    markets = dp.get_filtered_markets()

    assert all(~markets["name"].str.contains("USDT"))


def test_get_filtered_markets_names_should_not_contain_non_usd_fiat_currencies(dp):
    markets = dp.get_filtered_markets()

    for curr in constants.NON_USD_FIAT_CURRENCIES:
        assert all(~markets["name"].str.contains(curr))


def test_get_filtered_markets_should_not_contain_tokenized_equities(dp):
    markets = dp.get_filtered_markets()

    assert not any(markets["tokenizedEquity"] == True)  # "tokenizedEquity" could be nan


def test_get_filtered_markets_should_only_contain_equities_with_strictly_positive_volumes(
    dp,
):
    markets = dp.get_filtered_markets()

    assert all(markets["volumeUsd24h"] > 0)
