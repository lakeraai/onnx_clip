import pytest

from lakera_clip import Tokenizer


def test_text_too_long():
    text = [str(i) for i in range(100)]
    t = Tokenizer()
    with pytest.raises(RuntimeError):
        t.encode_text("".join(text))
