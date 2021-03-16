import numpy as np
import utils
from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes, from_dtype


@given(logit=st.floats(-20, 20))
def test_roundtrip_logit_to_p_p_to_logit(logit):
    value0 = utils.logit_to_p(logit=logit)
    value1 = utils.p_to_logit(p=value0)
    np.testing.assert_almost_equal(logit, value1)


def test_get_symbol():
    pass
