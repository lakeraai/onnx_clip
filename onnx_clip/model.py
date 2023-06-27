import errno
import os
import logging
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Iterator, TypeVar, Optional
import requests

import numpy as np
import onnxruntime as ort
from PIL import Image

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
    `get_similarity_scores(image_embeddings, text_embeddings)`.

    Args:
        embeddings_1: An array of embeddings of shape (N, D) or (D,).
        embeddings_2: An array of embeddings of shape (M, D) or (D,).

    Returns:
        An array with the pairwise similarity scores. If both inputs are 2-D,
            the output will be of shape (N, M). If one input is 1-D, the output
            will be of shape (N,) or (M,). If both inputs are 1-D, the output
            will be a scalar.
    """
    if embeddings_1.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity_scores(
            embeddings_1[np.newaxis, :], embeddings_2
        )[0]

    if embeddings_2.ndim == 1:
        # Convert to 2-D array using x[np.newaxis, :]
        # and remove the extra dimension at the end.
        return get_similarity_scores(
            embeddings_1, embeddings_2[np.newaxis, :]
        )[:, 0]

    return cosine_similarity(embeddings_1, embeddings_2) * 100


class OnnxClip:
    """
    This class can be utilised to predict the most relevant text snippet, given
    an image, without directly optimizing for the task, similarly to the
    zero-shot capabilities of GPT-2 and 3. The difference between this class
    and [CLIP](https://github.com/openai/CLIP) is that here we don't depend on
    `torch` or `torchvision`.
    """


    def __init__(
        self, model: str = "ViT-B/32", batch_size: Optional[int] = None, silent_download: bool = False
    ):
        """
        Instantiates the model and required encoding classes.

        Args:
            model: The model to utilise. Currently ViT-B/32 and RN50 are
                allowed.
            batch_size: If set, splits the lists in `get_image_embeddings`
                and `get_text_embeddings` into batches of this size before
                passing them to the model. The embeddings are then concatenated
                back together before being returned. This is necessary when
                passing large amounts of data (perhaps ~100 or more).
            silent_download: If True, the function won't show a warning in
                case when the models need to be downloaded from the S3 bucket.
        """
        allowed_models = ["ViT-B/32", "RN50"]
        if model not in allowed_models:
            raise ValueError(f"`model` must be in {allowed_models}. Got {model}.")
        if model == "ViT-B/32":
            self.embedding_size = 512
        elif model == "RN50":
            self.embedding_size = 1024
        self.image_model, self.text_model = self._load_models(model, silent_download)
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()
        self._batch_size = batch_size
    
    @property
    def EMBEDDING_SIZE(self):
        raise RuntimeError("OnnxModel.EMBEDDING_SIZE is no longer supported, please use the instance attribute: onnx_model.embedding_size")


    @staticmethod
    def _load_models(
        model: str,
        silent: bool,
    ) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """
        Grabs the ONNX implementation of CLIP's model :
        https://github.com/openai/CLIP/blob/main/clip/model.py

        We have exported it to ONNX to remove the dependency on `torch` and
        `torchvision`.
        """
        if model == "ViT-B/32":
            IMAGE_MODEL_FILE = "clip_image_model_vitb32.onnx"
            TEXT_MODEL_FILE = "clip_text_model_vitb32.onnx"
        elif model == "RN50":
            IMAGE_MODEL_FILE = "clip_image_model_rn50.onnx"
            TEXT_MODEL_FILE = "clip_text_model_rn50.onnx"
        else:
            raise ValueError(f"Unexpected model {model}. No `.onnx` file found.")
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
                # `providers` need to be set explicitly since ORT 1.9
                return ort.InferenceSession(
                    path, providers=ort.get_available_providers()
                )
            else:
                raise FileNotFoundError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    path,
                )
        except Exception:
            s3_url = f"https://lakera-clip.s3.eu-west-1.amazonaws.com/{os.path.basename(path)}"
            if not silent:
                logging.info(
                    f"The model file ({path}) doesn't exist "
                    f"or it is invalid. Downloading it from the public S3 "
                    f"bucket: {s3_url}."  # noqa: E501
                )

            # Download from S3 
            # Saving to a temporary file first to avoid corrupting the file
            temporary_filename = Path(path).with_name(os.path.basename(path) + '.part')
            
            # Create any missing directories in the path
            temporary_filename.parent.mkdir(parents=True, exist_ok=True)

            with requests.get(s3_url, stream=True) as r:
                r.raise_for_status()
                with open(temporary_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
                    f.flush()
            # Finally move the temporary file to the correct location
            temporary_filename.rename(path)

            # `providers` need to be set explicitly since ORT 1.9
            return ort.InferenceSession(
                path, providers=ort.get_available_providers()
            )

    def get_image_embeddings(
        self,
        images: Iterable[Union[Image.Image, np.ndarray]],
        with_batching: bool = True,
    ) -> np.ndarray:
        """Compute the embeddings for a list of images.

        Args:
            images: A list of images to run on. Each image must be a 3-channel
                (RGB) image. Can be any size, as the preprocessing step will
                resize each image to size (224, 224).
            with_batching: Whether to use batching - see the `batch_size` param
                in `__init__()`

        Returns:
            An array of embeddings of shape (len(images), embedding_size).
        """
        if not with_batching or self._batch_size is None:
            # Preprocess images
            images = [
                self._preprocessor.encode_image(image) for image in images
            ]
            if not images:
                return self._get_empty_embedding()

            batch = np.concatenate(images)

            return self.image_model.run(None, {"IMAGE": batch})[0]

        else:
            embeddings = []
            for batch in to_batches(images, self._batch_size):
                embeddings.append(
                    self.get_image_embeddings(batch, with_batching=False)
                )

            if not embeddings:
                return self._get_empty_embedding()

            return np.concatenate(embeddings)

    def get_text_embeddings(
        self, texts: Iterable[str], with_batching: bool = True
    ) -> np.ndarray:
        """Compute the embeddings for a list of texts.

        Args:
            texts: A list of texts to run on. Each entry can be at most
                77 characters.
            with_batching: Whether to use batching - see the `batch_size` param
                in `__init__()`

        Returns:
            An array of embeddings of shape (len(texts), embedding_size).
        """
        if not with_batching or self._batch_size is None:
            text = self._tokenizer.encode_text(texts)
            if len(text) == 0:
                return self._get_empty_embedding()

            return self.text_model.run(None, {"TEXT": text})[0]
        else:
            embeddings = []
            for batch in to_batches(texts, self._batch_size):
                embeddings.append(
                    self.get_text_embeddings(batch, with_batching=False)
                )

            if not embeddings:
                return self._get_empty_embedding()

            return np.concatenate(embeddings)

    def _get_empty_embedding(self):
        return np.empty((0, self.embedding_size), dtype=np.float32)


T = TypeVar("T")


def to_batches(items: Iterable[T], size: int) -> Iterator[List[T]]:
    """
    Splits an iterable (e.g. a list) into batches of length `size`. Includes
    the last, potentially shorter batch.

    Examples:
        >>> list(to_batches([1, 2, 3, 4], size=2))
        [[1, 2], [3, 4]]
        >>> list(to_batches([1, 2, 3, 4, 5], size=2))
        [[1, 2], [3, 4], [5]]

        # To limit the number of batches returned
        # (avoids reading the rest of `items`):
        >>> import itertools
        >>> list(itertools.islice(to_batches([1, 2, 3, 4, 5], size=2), 1))
        [[1, 2]]

    Args:
        items: The iterable to split.
        size: How many elements per batch.
    """
    if size < 1:
        raise ValueError("Chunk size must be positive.")

    batch = []
    for item in items:
        batch.append(item)

        if len(batch) == size:
            yield batch
            batch = []

    # The last, potentially incomplete batch
    if batch:
        yield batch
