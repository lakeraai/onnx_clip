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


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes softmax values for each sets of scores in x.
    This ensures the output sums to 1 for each image (along axis 1).
    """

    # Exponents
    exp_arr = np.exp(x)

    return exp_arr / np.sum(exp_arr, axis=1, keepdims=True)


def cosine_similarity(
    embeddings_1: np.ndarray, embeddings_2: np.ndarray
) -> np.ndarray:
    """Compute the pairwise cosine similarities between two embedding arrays.

    Args:
        embeddings_1: An array of embeddings of shape (N, D).
        embeddings_2: An array of embeddings of shape (M, D).

    Returns:
        An array of shape (N, M) with the pairwise cosine similarities.
    """

    for embeddings in [embeddings_1, embeddings_2]:
        if len(embeddings.shape) != 2:
            raise ValueError(
                f"Expected 2-D arrays but got shape {embeddings.shape}."
            )

    d1 = embeddings_1.shape[1]
    d2 = embeddings_2.shape[1]
    if d1 != d2:
        raise ValueError(
            "Expected second dimension of embeddings_1 and embeddings_2 to "
            f"match, but got {d1} and {d2} respectively."
        )

    def normalize(embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    embeddings_1 = normalize(embeddings_1)
    embeddings_2 = normalize(embeddings_2)

    return embeddings_1 @ embeddings_2.T


def get_similarity_scores(
    embeddings_1: np.ndarray, embeddings_2: np.ndarray
) -> np.ndarray:
    """Compute pairwise similarity scores between two arrays of embeddings.

    For zero-shot classification, these can be used as logits. To do so, call
    `get_similarity_scores(image_embeddings, text_embeddigs)`.

    Args:
        embeddings_1: An array of embeddings of shape (N, D).
        embeddings_2: An array of embeddings of shape (M, D).

    Returns:
        An array of shape (N, M) with the pairwise similarity scores.
    """
    return cosine_similarity(embeddings_1, embeddings_2) * 100


class OnnxClip:
    """
    This class can be utilised to predict the most relevant text snippet, given
    an image, without directly optimizing for the task, similarly to the
    zero-shot capabilities of GPT-2 and 3. The difference between this class
    and [CLIP](https://github.com/openai/CLIP) is that here we don't depend on
    `torch` or `torchvision`.
    """

    def __init__(self, silent_download: bool = False):
        """
        Instantiates the model and required encoding classes.

        Args:
            silent_download - if True, the function won't show a warning in
                case when the models need to be downloaded from the S3 bucket.
        """
        self.image_model, self.text_model = self._load_models(silent_download)
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    @staticmethod
    def _load_models(
        silent: bool,
    ) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """
        Grabs the ONNX implementation of CLIP's ViT-B/32 :
        https://github.com/openai/CLIP/blob/main/clip/model.py

        We have exported it to ONNX to remove the dependency on `torch` and
        `torchvision`.
        """
        IMAGE_MODEL_FILE = "clip_image_model_vitb32.onnx"
        TEXT_MODEL_FILE = "clip_text_model_vitb32.onnx"
        base_dir = os.path.dirname(os.path.abspath(__file__))

        models = []
        for model_file in [IMAGE_MODEL_FILE, TEXT_MODEL_FILE]:
            path = os.path.join(base_dir, "data", model_file)
            models.append(OnnxClip._load_model(path, silent))

        return models[0], models[1]

    @staticmethod
    def _load_model(path: str, silent: bool):
        try:
            if os.path.exists(path):
                return ort.InferenceSession(path)
            else:
                raise FileNotFoundError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    path,
                )
        except Exception:
            if not silent:
                logging.info(
                    f"The model file ({path}) doesn't exist "
                    f"or it is invalid. Downloading it from the public S3 "
                    f"bucket: https://lakera-clip.s3.eu-west-1.amazonaws.com/{os.path.basename(path)}."  # noqa: E501
                )
            # Download from S3
            s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
            s3_client.download_file(
                "lakera-clip", os.path.basename(path), path
            )
            return ort.InferenceSession(path)

    def get_image_embeddings(
        self, images: Union[List[Image.Image], List[np.ndarray]]
    ) -> np.ndarray:
        """Compute the embeddings for a list of images.

        Args:
            images: A list of images to run on. Each image must be a 3-channel
                (RGB) image. Can be any size, as the preprocessing step will
                resize each image to size (224, 224).

        Returns:
            An array of embeddings of shape (len(images), 512).
        """
        # Preprocess images
        images = [self._preprocessor.encode_image(image) for image in images]
        # Concatenate
        batch = np.concatenate(images)

        return self.image_model.run(None, {"IMAGE": batch})[0]

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute the embeddings for a list of texts.

        Args:
            texts: A list of texts to run on. Each entry can be at most
                77 characters.

        Returns:
            An array of embeddings of shape (len(texts), 512).
        """
        text = self._tokenizer.encode_text(texts)
        return self.text_model.run(None, {"TEXT": text})[0]
