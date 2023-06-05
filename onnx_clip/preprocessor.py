from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image


class Preprocessor:
    """
    Our approach to the CLIP `preprocess` neural net that does not rely on PyTorch.
    The two approaches fully match.
    """

    # Fixed variables that ensure the correct output shapes and values for the `Model` class.
    CLIP_INPUT_SIZE = 224
    # Normalization constants taken from original CLIP:
    # https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L85
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(
        (1, 1, 3)
    )
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(
        (1, 1, 3)
    )

    @staticmethod
    def _crop_and_resize(img: np.ndarray) -> np.ndarray:
        """Resize and crop an image to a square, preserving the aspect ratio."""

        # Current height and width
        h, w = img.shape[0:2]

        if h * w == 0:
            raise ValueError(
                f"Height and width of the image should both be non-zero but got shape {h, w}"
            )

        target_size = Preprocessor.CLIP_INPUT_SIZE

        # Resize so that the smaller dimension matches the required input size.
        # Matches PyTorch:
        # https://github.com/pytorch/vision/blob/7cf0f4cc1801ff1892007c7a11f7c35d8dfb7fd0/torchvision/transforms/functional.py#L366
        if h < w:
            resized_h = target_size
            resized_w = int(resized_h * w / h)
        else:
            resized_w = target_size
            resized_h = int(resized_w * h / w)

        # PIL resizing behaves slightly differently than OpenCV because of
        # antialiasing. See also
        # https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html
        # CLIP uses PIL, so we do too to match its results. But if you don't
        # want to have PIL as a dependency, feel free to change the code to
        # use the other branch.
        use_pil_for_resizing = True

        if use_pil_for_resizing:
            # https://github.com/pytorch/vision/blob/7cf0f4cc1801ff1892007c7a11f7c35d8dfb7fd0/torchvision/transforms/functional_pil.py#L240
            # We're working with float images but PIL uses uint8, so convert
            # there and back again afterwards
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize(
                (resized_w, resized_h), resample=Image.BICUBIC
            )
            img = np.array(img_pil).astype(np.float32) / 255
        else:
            img = cv.resize(
                img, (resized_w, resized_h), interpolation=cv.INTER_CUBIC
            )

        # Now crop to a square
        y_from = (resized_h - target_size) // 2
        x_from = (resized_w - target_size) // 2
        img = img[
            y_from : y_from + target_size, x_from : x_from + target_size, :
        ]

        return img

    @staticmethod
    def _image_to_float_array(img: Union[Image.Image, np.ndarray]):
        """Converts a PIL image or a NumPy array to standard form.

        Standard form means:
        - the shape is (H, W, 3)
        - the dtype is np.float32
        - all values are in [0, 1]
        - there are no NaN values

        Args:
            img: The image to convert.

        Returns:
            The image converted to a NumPy array in standard form.

        Raises:
            ValueError if the image is invalid (wrong shape, invalid
                values...).
        """
        if not isinstance(img, (Image.Image, np.ndarray)):
            raise TypeError(
                f"Expected PIL Image or np.ndarray but instead got {type(img)}"
            )

        if isinstance(img, Image.Image):
            # Convert to NumPy
            img = np.array(img)

        if len(img.shape) > 3:
            raise ValueError(
                f"The image should have 2 or 3 dimensions but got "
                f"{len(img.shape)} dimensions"
            )
        if len(img.shape) == 3 and img.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel RGB image but got image with "
                f"{img.shape[2]} channels"
            )

        # Handle grayscale
        if len(img.shape) == 2:
            # The model doesn't support HxWx1 images as input
            img = np.expand_dims(img, axis=2)  # HxWx1
            img = np.concatenate((img,) * 3, axis=2)  # HxWx3

        # At this point, `img` has the shape (H, W, 3).

        if np.min(img) < 0:
            raise ValueError(
                "Images should have non-negative pixel values, "
                f"but the minimum value is {np.min(img)}"
            )

        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) > 1:
                raise ValueError(
                    "Images with a floating dtype should have values "
                    f"in [0, 1], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32)
        elif np.issubdtype(img.dtype, np.integer):
            if np.max(img) > 255:
                raise ValueError(
                    "Images with an integer dtype should have values "
                    f"in [0, 255], but the maximum value is {np.max(img)}"
                )
            img = img.astype(np.float32) / 255
            img = np.clip(img, 0, 1)  # In case of rounding errors
        else:
            raise ValueError(
                f"The image has an unsupported dtype: {img.dtype}."
            )

        if np.isnan(img).any():
            raise ValueError(f"The image contains NaN values.")

        try:
            # These should never trigger, but let's do a sanity check
            assert np.min(img) >= 0
            assert np.max(img) <= 1
            assert img.dtype == np.float32
            assert len(img.shape) == 3
            assert img.shape[2] == 3
        except AssertionError as e:
            raise RuntimeError(
                "Internal preprocessing error. "
                "The image does not have the expected format."
            ) from e

        return img

    def encode_image(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocesses the images like CLIP's preprocess() function:
        https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L79

        Args:
            img: PIL image or numpy array

        Returns:
            img: numpy image after resizing, center cropping and normalization.
        """
        img = Preprocessor._image_to_float_array(img)

        img = Preprocessor._crop_and_resize(img)

        # Normalize channels
        img = (img - Preprocessor.NORM_MEAN) / Preprocessor.NORM_STD

        # Mimic the pytorch tensor format for Model class
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img
