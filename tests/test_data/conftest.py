import pytest
from algtra import data
from ftx.rest.client import FtxClient
import keys
import os


@pytest.fixture
def dp():
    return data.DataProcessor()


@pytest.fixture
def pdp():
    return data.PriceDataProcessor()


@pytest.fixture
def client():
    return FtxClient(keys.ftx_api_key, keys.ftx_secret_key)


@pytest.fixture
def test_coins():
    return ["BTC", "FTT", "ADA"]


@pytest.fixture
def data_dir(test_coins):

    yield os.getcwd()
