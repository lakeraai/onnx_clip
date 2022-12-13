import os

import numpy as np
import pytest
from PIL import Image

from onnx_clip import OnnxClip, softmax


def load_image_text(convert=True):
    """
    Load a test image and convert to 3-channel RBG instead of 4-channel RGBA.

    Returns:
        (test_image, test_text)
    """
    IMAGE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../onnx_clip/data/CLIP.png"
    )
    if convert:
        return Image.open(IMAGE_PATH).convert("RGB"), [
            "a photo of a man",
            "a photo of a woman",
        ]
    else:
        return Image.open(IMAGE_PATH), [
            "a photo of a man",
            "a photo of a woman",
        ]


def test_bad_image_input():
    """
    Test that a non-PIL input is bad for an image.
    """
    _, text = load_image_text()

    onnx_model = OnnxClip()
    with pytest.raises(TypeError):
        onnx_model.predict("bad image input", text)


def test_bad_image_channels():
    """
    Test that a 4-channel image raises the appropriate error.
    """
    image, text = load_image_text(convert=False)

    onnx_model = OnnxClip()
    with pytest.raises(ValueError):
        onnx_model.predict([image], text)


def test_bad_text_input():
    """
    Test that a non-tokenized input text is bad for model.
    """
    image, _ = load_image_text()

    onnx_model = OnnxClip()
    with pytest.raises(TypeError):
        onnx_model.predict([image], 123)


def test_softmax_values():
    """
    Test the softmax function works as expected.
    """

    logits = np.array([[0, 10, -10]])
    assert all(np.isclose(np.sum(softmax(logits), axis=1), 1))


def test_model_runs():
    """
    Test full process.
    """
    image, text = load_image_text()
    n_images = 3
    n_text = 2

    onnx_model = OnnxClip()

    logits_per_image, logits_per_text = onnx_model.predict(
        images=n_images * [image],
        text=text
    )

    assert logits_per_image.shape == (n_images, n_text)
    assert logits_per_text.shape == (n_text, n_images)

    probas = softmax(logits_per_image)

    assert probas.shape == (n_images, n_text)

    # values taken from pytorch with ViT-B/32
    probs_clip = [0.6846084, 0.31539157]

    assert abs(probs_clip[0] - probas[0][0]) <= 5
    assert abs(probs_clip[1] - probas[0][1]) <= 5
