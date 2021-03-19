import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from algtra import utils
from algtra.collect import data
from algtra.constants import PARTIAL_LEVERAGED_SYMBOLS
from ftx.rest.client import FtxClient
import keys
from hypothesis import given, assume, example, strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype


@given(logit=st.floats(-20, 20))
def test_roundtrip_logit_to_p_p_to_logit(logit):
    value0 = utils.logit_to_p(logit=logit)
    value1 = utils.p_to_logit(p=value0)
    np.testing.assert_almost_equal(logit, value1)


class Test_get_symbol_and_get_coin:
    def test_get_coin(self):
        client = FtxClient(keys.ftx_api_key, keys.ftx_secret_key)
        markets = data.get_filtered_markets(client)

        for curr in markets["baseCurrency"]:
            coin = utils.get_coin(curr)
            for partial_leveraged_symbol in PARTIAL_LEVERAGED_SYMBOLS:
                assert partial_leveraged_symbol not in coin

    @given(s=st.text(min_size=1))
    @example("BULL")
    @example("BEAR")
    @example("HEDGE")
    @example("HALF")
    def test_get_coin_is_fixed_point(self, s):
        assume(s is not None)

        s = utils.get_coin(s)
        assert utils.get_coin(s) == s

    @given(s=st.text(min_size=1), m=st.sampled_from([1, -1, 3, -3]))
    @example(s="BTC", m=1)
    @example(s="BTC", m=-1)
    @example(s="BTC", m=3)
    @example(s="BTC", m=-3)
    def test_roundtrip_get_symbol_get_coin(self, s, m):
        s0 = utils.get_symbol(s, m=np.abs(m), bear=m < 0)
        assert s0 != s or m == 1
        s1 = utils.get_coin(s0)
        assert s1 == s
