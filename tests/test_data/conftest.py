import pytest
from algtra import data
from ftx.rest.client import FtxClient
import keys


@pytest.fixture
def dp():
    return data.DataProcessor()


@pytest.fixture
def pdp():
    return data.PriceDataProcessor()


@pytest.fixture
def client():
    return FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
