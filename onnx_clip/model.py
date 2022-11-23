import errno
import os
from typing import List, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from onnx_clip import Preprocessor, Tokenizer


class OnnxClip:
    """
    This class utilises both the Tokenizer and Preprocess classes to encode the text and images alongside the ONNX
    format of the model.
    This is done under the hood to allow for ease of code.

    Example usage:
        image = Image.open("lakera_clip/data/CLIP.png")
        text = ["a photo of a man", "a photo of a woman"]
        onnx_clip = OnnxClip()
        logits_per_image, logits_per_text = onnx_clip.run(image, text)
        probas = lakera_model.softmax(logits_per_image)
    """

    def __init__(self):
        """
        Instantiates the model and required encoding classes.
        """
        self.model = self._load_model()
        self._tokenizer = Tokenizer()
        self._preprocessor = Preprocessor()

    def _load_model(self):
        """
        Grabs the ONNX model.
        """
        MODEL_ONNX_EXPORT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/clip_model.onnx"
        )
        if os.path.exists(MODEL_ONNX_EXPORT_PATH):
            return ort.InferenceSession(MODEL_ONNX_EXPORT_PATH)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), MODEL_ONNX_EXPORT_PATH
            )

    def run(
        self, image: Image.Image, text: Union[str, List[str]]
    ) -> Tuple[np.array, np.array]:
        """
        Calculates the logits. Both the Tokenizer and Preprocess classes are used to encode
        the text and image respectively.

        Args:
            image: the original PIL image
            text: the text to tokenize

        Returns:
            (logits_per_image, logits_per_text) tuple.
        """
        image = self._preprocessor.encode_image(image)
        text = self._tokenizer.encode_text(text)

        logits_per_image, logits_per_text = self.model.run(
            None, {"IMAGE": image, "TEXT": text}
        )
        return logits_per_image, logits_per_text

    def softmax(self, x: np.array) -> np.array:
        """
        Computes softmax values for each sets of scores in x.
        This ensures the output sums to 1.
        """
        return (np.exp(x) / np.sum(np.exp(x), axis=1))[0]
