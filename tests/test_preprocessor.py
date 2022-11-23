import pytest

from onnx_clip import Preprocessor


def test_bad_input_type():
    pre = Preprocessor()
    with pytest.raises(TypeError):
        pre.encode_image("this should raise an error")
