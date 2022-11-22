import pytest

from lakera_clip import Preprocess


def test_bad_input_type():
    pre = Preprocess()
    with pytest.raises(AssertionError):
        pre.encode_image("this should raise an error")
