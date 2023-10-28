from __future__ import annotations

import os

import importlib_metadata

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_package_available(pkg_name: str) -> bool:
    try:
        package_version = importlib_metadata.metadata(pkg_name) is not None
        return bool(package_version)
    except importlib_metadata.PackageNotFoundError:
        return False


_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = _is_package_available("torch")
else:
    _torch_available = False

_transformers_available = _is_package_available("transformers")


def is_torch_available():
    return _torch_available


def is_transformers_available():
    return _transformers_available


is_cuda_pipline_available = is_torch_available() and is_transformers_available()
