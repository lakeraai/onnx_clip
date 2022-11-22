import cv2 as cv
import numpy as np
from PIL import Image


class Preprocess:
    def __init__(self):
        self.CLIP_INPUT_SIZE = 224
        self.CROP_CENTER_PADDING = 224
        self.NORM_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.NORM_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def _smart_resize(self, img: Image.Image) -> np.array:
        """Resizing that preserves the image ratio"""
        # The expected size of the image after we resize it
        # and pad to have a square format
        resized_sq_size = self.CLIP_INPUT_SIZE + 2 * self.CROP_CENTER_PADDING
        # Current height and width
        img = np.array(img)
        h, w = img.shape[0:2]
        # The size of the image after we resize but before we pad
        if h > w:
            resized_h = resized_sq_size
            resized_w = round(resized_h * w / h)
        else:
            resized_w = resized_sq_size
            resized_h = round(resized_w * h / w)

        # Resize preserving the ratio
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

    def _assert_pil(self, img: Image.Image):
        if not isinstance(img, Image.Image):
            raise AssertionError(f"Expected PIL Image but instead got {type(img)}")

    def encode_image(self, img: Image.Image) -> np.array:
        """
        The function for preprocessing the images in the same style as CLIP's preprocess() function.
        Through experimentation, the best method seems to be a carbon-copy of ConvNextExtractor found in
        lakera/internal/embedding_extractor.py
        Args:
            img: PIL image

        Returns:
            img: numpy image after resizing, interpolation and center cropping.

        """
        self._assert_pil(img)
        # Resize
        img = self._smart_resize(img)
        # Crop the center
        img = img[
            self.CROP_CENTER_PADDING : -self.CROP_CENTER_PADDING,
            self.CROP_CENTER_PADDING : -self.CROP_CENTER_PADDING,
        ]

        # [0; 255] -> [0; 1]
        img = img / 255.0
        # Handle Grayscale
        if len(img.shape) == 2:
            # The NN doesn't support NxMx1 images as input
            img = np.expand_dims(img, axis=2)  # NxMx1
            img = np.concatenate((img,) * 3, axis=2)  # NxMx3
        # Normalize
        img = (img - self.NORM_MEAN) / self.NORM_STD
        # To tensor-friendly format
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32).reshape(1, 3, 224, 224)

        return img
