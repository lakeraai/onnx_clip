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


def test_bad_inputs():
    pre = Preprocessor()
    with pytest.raises(TypeError):
        pre.encode_image("this should raise an error")

    with pytest.raises(ValueError):
        pre.encode_image(np.zeros((10, 10, 3), dtype=object))

    with pytest.raises(ValueError):
        pre.encode_image(np.zeros((10, 10, 1)))

    with pytest.raises(ValueError):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[2, 2, 0] = np.nan
        pre.encode_image(img)

    with pytest.raises(ValueError):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[2, 2, 0] = 1.00001
        pre.encode_image(img)

    with pytest.raises(ValueError):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[2, 2, 0] = -0.00001
        pre.encode_image(img)

    with pytest.raises(ValueError):
        img = np.zeros((10, 10, 3), dtype=int)
        img[2, 2, 0] = 256
        pre.encode_image(img)

    with pytest.raises(ValueError):
        img = np.zeros((10, 10, 3), dtype=int)
        img[2, 2, 0] = -1
        pre.encode_image(img)


#
# pre.encode_image(np.zeros((100, 1, 1)))


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


def test_dtypes():
    image = Image.open(IMAGE_PATH).convert("RGB")
    pre = Preprocessor()

    image_np_uint8 = np.array(image)
    assert image_np_uint8.dtype == np.uint8

    image_np_float = image_np_uint8.astype(np.float32) / 255.0
    image_np_int = image_np_uint8.astype(int)

    prepped_image = pre.encode_image(image)
    prepped_image_np = pre.encode_image(image_np_uint8)
    prepped_image_np_float = pre.encode_image(image_np_float)
    prepped_image_np_int = pre.encode_image(image_np_int)

    assert np.allclose(prepped_image, prepped_image_np)
    assert np.allclose(prepped_image, prepped_image_np_float)
    assert np.allclose(prepped_image, prepped_image_np_int)


def test_special_shapes():
    pre = Preprocessor()

    pre.encode_image(np.zeros((100, 1, 3)))  # width 1
    pre.encode_image(np.zeros((100, 50)))  # grayscale
