import onnxruntime as ort
import os
import errno
import numpy as np
from PIL import Image

class preprocess():
    def __init__(self):
        self.onnx_preprocessor = self._load_preprocessor()
        self.ONNX_INPUT_NAMES = ["IMAGE"]
        self.ONNX_OUTPUT_NAMES = ["PREPROCESSED_IMAGE"]


    def _load_preprocessor(self):
        PREPROCESSOR_ONNX_EXPORT_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/clip_preprocessor.onnx"
        )
        if os.path.exists(PREPROCESSOR_ONNX_EXPORT_PATH):
            return ort.InferenceSession(PREPROCESSOR_ONNX_EXPORT_PATH)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), PREPROCESSOR_ONNX_EXPORT_PATH
            )

    def _assert_pil(self, image):
        if not isinstance(image, Image.Image):
            raise AssertionError(f"Expected PIL Image but instead got {image.type}")

    def run(self, image):
        """
        todo: must expect a PIL image
        """
        self._assert_pil(image)
        image = self._pil_to_numpy(image)
        image = self._transpose(image)
        return self.onnx_preprocessor.run(self.ONNX_OUTPUT_NAMES, {self.ONNX_INPUT_NAMES[0]: image})[0].reshape(1, 3, 224, 224)


    def _pil_to_numpy(self, image):
        return np.array(image, dtype=np.float32)

    def _transpose(self, array):
        return np.transpose(array, (2, 0, 1)) / 255.