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
    # Normalization constants taken from original CLIP:
    # https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L85
    NORM_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(
        (1, 1, 3)
    )
    NORM_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(
        (1, 1, 3)
    )

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
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(
                (resized_w, resized_h), resample=Image.BICUBIC
            )
            img = np.array(img_pil)
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
            raise TypeError(
                f"Expected PIL Image or np.ndarray but instead got {type(img)}"
            )

        if isinstance(img, Image.Image):
            # Convert to NumPy
            img = np.array(img)

        # Resize
        img = self._smart_resize(img)

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
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)

        return img
