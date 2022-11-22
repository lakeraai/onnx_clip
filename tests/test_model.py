import os

import numpy as np
import pytest
from PIL import Image

from lakera_clip import Model


def load_image_text():
    """
    Load a test image and convert to 3-channel RBG instead of 4-channel RGBA.

    Returns:
        (test_image, test_text)
    """
    IMAGE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../lakera_clip/data/CLIP.png"
    )
    return Image.open(IMAGE_PATH).convert("RGB"), [
        "a photo of a man",
        "a photo of a woman",
    ]


def test_bad_image_input():
    """
    Test that a non-PIL input is bad for an image.
    """
    _, text = load_image_text()

    onnx_model = Model()
    with pytest.raises(AssertionError):
        onnx_model.run("bad image input", text)


def test_bad_text_input():
    """
    Test that a non-tokenized input text is bad for model.
    """
    image, _ = load_image_text()

    onnx_model = Model()
    with pytest.raises(TypeError):
        onnx_model.run(image, 123)


def test_softmax_values():
    """
    Test the softmax function works as expected.
    """
    onnx_model = Model()
    logits = np.array([[0, 10, -10]])
    assert sum(onnx_model.softmax(logits)) == 1


def test_model_runs():
    """
    Test full process.
    """
    image, text = load_image_text()

    onnx_model = Model()

    logits_per_image, logits_per_text = onnx_model.run(image, text)

    assert logits_per_image.shape == (1, 2)
    assert logits_per_text.shape == (2, 1)

    probas = onnx_model.softmax(logits_per_image)

    assert len(probas) == 2
