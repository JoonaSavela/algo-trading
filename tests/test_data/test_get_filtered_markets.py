def test_get_filtered_markets_should_not_contain_USDT(dp):
    markets = dp.get_filtered_markets()

    assert all()
