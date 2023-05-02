import importlib_metadata

from .preprocessor import Preprocessor
from .tokenizer import Tokenizer
from .model import OnnxClip, softmax, get_similarity_scores

__all__ = [
    "Preprocessor",
    "Tokenizer",
    "OnnxClip",
    "softmax",
    "get_similarity_scores",
]

# Matches the version specified pyproject.toml under [tool.poetry]
__version__ = importlib_metadata.version("onnx_clip")
