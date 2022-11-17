import os
import errno
import onnxruntime as ort
from PIL import Image
import numpy as np

class model():
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        MODEL_ONNX_EXPORT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/clip_model.onnx"
        )
        if os.path.exists(MODEL_ONNX_EXPORT_PATH):
            return ort.InferenceSession(MODEL_ONNX_EXPORT_PATH)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), MODEL_ONNX_EXPORT_PATH
            )

    def _assert_pil(self, image):
        if not isinstance(image, Image.Image):
            raise AssertionError(f"Expected PIL Image but instead got {image.type}")

    def run(self, image, text):
        logits_per_image, logits_per_text = self.model.run(
            None, {"IMAGE": image, "TEXT": text}
        )
        return logits_per_image, logits_per_text

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return (np.exp(x) / np.sum(np.exp(x), axis=1))[0]