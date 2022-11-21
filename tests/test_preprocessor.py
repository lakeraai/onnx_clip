import pytest

from lakera_clip import preprocess
def test_bad_input_type():
    pre = preprocess()
    with pytest.raises(AssertionError):
        pre.encode_image("this should raise an error")

