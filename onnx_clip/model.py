import errno
import os
import logging
from typing import List, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from onnx_clip import Preprocessor, Tokenizer


def softmax(x: np.array) -> np.array:
    """
    Computes softmax values for each sets of scores in x.
    This ensures the output sums to 1 for each image (along axis 1).
    """

    # Exponents
    exp_arr = np.exp(x)

    return exp_arr / np.sum(exp_arr, axis=1, keepdims=True)


class OnnxClip:
    """
    This class can be utilised to predict the most relevant text snippet, given an image,
    without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.
    The difference between this class and [CLIP](https://github.com/openai/CLIP) is that here we do not use any
    PyTorch dependencies.
    """

    def __init__(self, silent_download: bool = False):
        """
        Instantiates the model and required encoding classes.

        Args:
            silent_download - if True, the function won't show a warning in
                case when the model needs to be downloaded from the S3 bucket.
        """
        self.model = self._load_model(silent_download)
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    def _load_model(self, silent: bool):
        """
        Grabs the ONNX implementation of CLIP's ViT-B/32 :
        https://github.com/openai/CLIP/blob/main/clip/model.py

        We have exported it to ONNX to remove the PyTorch dependencies.
        """
        MODEL_ONNX_EXPORT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data/clip_model_vitb32.onnx"
        )

        try:
            if os.path.exists(MODEL_ONNX_EXPORT_PATH):
                model = ort.InferenceSession(MODEL_ONNX_EXPORT_PATH)
            else:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT),
                    MODEL_ONNX_EXPORT_PATH
                )
        except Exception:
            if not silent:
                logging.warning(
                    f"The model file ({MODEL_ONNX_EXPORT_PATH}) doesn't exist "
                    f"or it is invalid. Downloading it from the public S3 "
                    f"bucket instead: https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_model.onnx."  # noqa: E501
                )
            # Download from S3
            s3_client = boto3.client(
                's3', config=Config(signature_version=UNSIGNED)
            )
            s3_client.download_file(
                'lakera-clip', 'clip_model.onnx', MODEL_ONNX_EXPORT_PATH
            )
            model = ort.InferenceSession(MODEL_ONNX_EXPORT_PATH)

        return model

    def predict(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        text: Union[str, List[str]]
    ) -> Tuple[np.array, np.array]:
        """
        Given a raw image and a list of text categories, returns two arrays, containing the logit scores corresponding
        to each image and text input.
        The values are cosine similarities between the corresponding image and text features, times 100.

        The images and text are encoded in a similar manner to the `preprocess` and `tokenize` functions within CLIP,
        after which they are passed to the ONNX version of the CLIP model.

        Example usage:
            from onnx_clip import OnnxClip, softmax
            from PIL import Image

            images = [Image.open("lakera_clip/data/CLIP.png").convert("RGB")]
            text = ["a photo of a man", "a photo of a woman"]

            onnx_model = OnnxClip()
            logits_per_image, logits_per_text = onnx_model.predict(images, text)
            probas = softmax(logits_per_image)

            print(logits_per_image, probas)
            [[20.380428 19.790262]], [[0.64340323 0.35659674]]

            print(logits_per_text)
            [
                [20.380428],
                [19.790262]
            ]

        Args:
            images: the original PIL image or numpy array. This image must be a 3-channel (RGB) image.
                Can be any size, as the preprocessing step is done to convert this image to size (224, 224).
            text: the text to tokenize. Each category in the given list cannot be longer than 77 characters.

        Returns:
            logits_per_image: The scaled dot product scores between each image embedding and the text embeddings.
                This represents the image-text similarity scores.
            logits_per_text: The scaled dot product scores between each text embedding and the image embeddings.
                This represents the text-image similarity scores.
        """
        # Preprocess images
        images = [self._preprocessor.encode_image(image) for image in images]
        # Concatenate
        batch = np.concatenate(images)
        # Preprocess text
        text = self._tokenizer.encode_text(text)

        logits_per_image, logits_per_text = self.model.run(
            None, {"IMAGE": batch, "TEXT": text}
        )
        return logits_per_image, logits_per_text
