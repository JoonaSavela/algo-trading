from algtra import constants


def test_get_coins_returns_a_list(dp):
    assert isinstance(dp.get_coins(), list)


def test_get_coins_contains_at_least_100_unique_items(dp):
    assert len(set(dp.get_coins())) >= 100


def test_coins_in_get_coins_do_not_contain_fiat_currencies(dp):
    coins = dp.get_coins()

    for curr in constants.NON_USD_FIAT_CURRENCIES:
        assert all(curr != coin for coin in coins)


def test_get_coins_is_a_list_of_uppercase_strings(dp):
    for coin in dp.get_coins():
        assert isinstance(coin, str)
        assert coin == coin.upper()
