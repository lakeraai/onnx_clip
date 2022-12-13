from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image


class Preprocessor:
    """
    A rough approximation to the CLIP `preprocess` neural net.
    """

    # Fixed variables that ensure the correct output shapes and values for the `Model` class.
    CLIP_INPUT_SIZE = 224
    CROP_CENTER_PADDING = 224
    NORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    NORM_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def _smart_resize(self, img: np.ndarray) -> np.array:
        """Resizing that preserves the image ratio"""

        if len(img.shape) > 3:
            raise ValueError(
                f"The image should have 2 or 3 dimensions but got {len(img.shape)} dimensions"
            )
        if (len(img.shape) == 3) and img.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image but got image with {img.shape[2]} channels"
            )

        # The expected size of the image after we resize it
        # and pad to have a square format
        resized_sq_size = (
            Preprocessor.CLIP_INPUT_SIZE + 2 * Preprocessor.CROP_CENTER_PADDING
        )

        # Current height and width
        h, w = img.shape[0:2]

        if h * w == 0:
            raise ValueError(
                f"Height and width of the image should both be non-zero but got shape {h, w}"
            )

        # The size of the image after we resize but before we pad
        if h > w:
            resized_h = resized_sq_size
            resized_w = round(resized_h * w / h)
        else:
            resized_w = resized_sq_size
            resized_h = round(resized_w * h / w)

        # Resize while preserving the ratio
        img = cv.resize(img, (resized_w, resized_h), interpolation=cv.INTER_LINEAR)

        # Pad the image to make it square
        vert_residual = resized_sq_size - resized_h
        hor_residual = resized_sq_size - resized_w
        vert_pad = vert_residual // 2
        hor_pad = hor_residual // 2
        if len(img.shape) == 3:
            padding = (
                (vert_pad, vert_residual - vert_pad),
                (hor_pad, hor_residual - hor_pad),
                (0, 0),
            )
        else:
            # If grayscale, cv.resize will drop the last dimension
            padding = (
                (vert_pad, vert_residual - vert_pad),
                (hor_pad, hor_residual - hor_pad),
            )
        img = np.pad(
            img,
            padding,
            constant_values=0,
        )
        return img

    def encode_image(self, img: Union[Image.Image, np.ndarray]) -> np.array:
        """
        The function for preprocessing the images in an approximate way to CLIP's preprocess() function:
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        This is the function that causes a (small) deviation from CLIP results, as the interpolation and
        normalization methodologies are different.
        Args:
            img: PIL image or numpy array

        Returns:
            img: numpy image after resizing, interpolation and center cropping.

        """
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(f"Expected PIL Image but instead got {type(img)}")

        if isinstance(img, Image.Image):
            # Convert to NumPy
            img = np.array(img)
        # Resize
        img = self._smart_resize(img)
        # Crop the center
        img = img[
            Preprocessor.CROP_CENTER_PADDING : -Preprocessor.CROP_CENTER_PADDING,
            Preprocessor.CROP_CENTER_PADDING : -Preprocessor.CROP_CENTER_PADDING,
        ]

        # Normalize to values [0, 1]
        img = img / 255.0

        # Handle Grayscale
        if len(img.shape) == 2:
            # The NN doesn't support NxMx1 images as input
            img = np.expand_dims(img, axis=2)  # NxMx1
            img = np.concatenate((img,) * 3, axis=2)  # NxMx3

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32).reshape(1, 3, 224, 224)

        return img
