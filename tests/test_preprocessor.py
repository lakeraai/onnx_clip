import os

import numpy as np
import pytest
from PIL import Image

from onnx_clip import Preprocessor

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../onnx_clip/data/",
)

IMAGE_PATH = os.path.join(DATA_DIR, "franz-kafka.jpg")
PREPROCESSED_IMAGE_PATH = os.path.join(
    DATA_DIR, "expected_preprocessed_image.npy"
)


def test_bad_input_type():
    pre = Preprocessor()
    with pytest.raises(TypeError):
        pre.encode_image("this should raise an error")


def _debug_preprocessing(expected_preprocessed_image, preprocessed_image):
    def show_image(img, is_original):
        x = np.moveaxis(img[0], 0, -1)
        x = (x * 255).astype(np.uint8)
        if is_original:
            x[:10, :10, :] = 0
        Image.fromarray(x).show()

    # For debugging
    show_image(expected_preprocessed_image, is_original=True)
    show_image(preprocessed_image, is_original=False)
    show_image(
        expected_preprocessed_image - preprocessed_image, is_original=False
    )
    print(np.max(preprocessed_image - expected_preprocessed_image))


def test_matches_original_clip():
    image = Image.open(IMAGE_PATH).convert("RGB")
    pre = Preprocessor()
    preprocessed_image = pre.encode_image(image)

    assert preprocessed_image.shape == (
        1,
        3,
        Preprocessor.CLIP_INPUT_SIZE,
        Preprocessor.CLIP_INPUT_SIZE,
    )

    expected_preprocessed_image = np.load(PREPROCESSED_IMAGE_PATH)

    # _debug_preprocessing(expected_preprocessed_image, preprocessed_image)

    assert np.allclose(
        preprocessed_image, expected_preprocessed_image, atol=1e-6
    )
