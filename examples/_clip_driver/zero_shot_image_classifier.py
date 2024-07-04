from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import onnxruntime
from PIL.Image import Image

from .moss_clip import MossCLIP


def register_pipline(*, visual_path: Path, textual_path) -> MossCLIP:
    v_net, t_net = None, None

    if visual_path:
        if not isinstance(visual_path, Path):
            raise ValueError("visual_path should be a pathlib.Path")
        if not visual_path.is_file():
            raise FileNotFoundError(
                f"Select to use visual ONNX model, but the specified model does not exist - {visual_path=}"
            )
        v_net = onnxruntime.InferenceSession(
            visual_path, providers=onnxruntime.get_available_providers()
        )
    if textual_path:
        if not isinstance(textual_path, Path):
            raise ValueError("textual_path should be a pathlib.Path")
        if not textual_path.is_file():
            raise FileNotFoundError(
                f"Select to use textual ONNX model, but the specified model does not exist - {textual_path=}"
            )
        t_net = onnxruntime.InferenceSession(
            textual_path, providers=onnxruntime.get_available_providers()
        )

    if not v_net or not t_net:
        raise ValueError("Model initialization failed")

    _pipeline = MossCLIP.from_pluggable_model(v_net, t_net)
    return _pipeline


def format_datalake(dl: DataLake) -> Tuple[List[str], List[str]]:
    positive_labels = dl.positive_labels.copy()
    negative_labels = dl.negative_labels.copy()

    # When the input is a challenge prompt, cut it into phrases
    if dl.raw_prompt:
        prompt = dl.raw_prompt
        true_label = prompt.replace("_", " ")
        if true_label not in positive_labels:
            positive_labels.append(true_label)
        if not negative_labels:
            false_label = f"This is a photo that has nothing to do with {true_label}."
            negative_labels.append(false_label)

    # Insert hypothesis_template
    for labels in [positive_labels, negative_labels]:
        for i, label in enumerate(labels):
            if "a photo" in label:
                continue
            labels[i] = f"This is a photo of the {label}."

    # Formatting model input
    candidate_labels = positive_labels.copy()
    if isinstance(negative_labels, list) and len(negative_labels) != 0:
        candidate_labels.extend(negative_labels)

    return positive_labels, candidate_labels


@dataclass
class ZeroShotImageClassifier:
    positive_labels: List[str] = field(default_factory=list)
    candidate_labels: List[str] = field(default_factory=list)

    @classmethod
    def from_datalake(cls, dl: DataLake):
        positive_labels, candidate_labels = format_datalake(dl)
        return cls(positive_labels=positive_labels, candidate_labels=candidate_labels)

    def __call__(self, detector: MossCLIP, image: Image, *args, **kwargs):
        if isinstance(detector, MossCLIP) and not isinstance(image, Iterable):
            image = [image]
        predictions = detector(image, candidate_labels=self.candidate_labels)
        return predictions


@dataclass
class DataLake:
    positive_labels: List[str] = field(default_factory=list)
    """
    Indicate the label with the meaning "True", 
    preferably an independent noun or clause
    """

    negative_labels: List[str] = field(default_factory=list)
    """
    Indicate the label with the meaning "False", 
    preferably an independent noun or clause
    """

    joined_dirs: List[str] | Path | None = None
    """
    Attributes reserved for AutoLabeling
    Used to indicate the directory where the dataset is located

    input_dir = db_dir.joinpath(*joined_dirs).absolute()
    """

    raw_prompt: str = ""
    """
    Challenge prompt or keywords after being divided

    !! IMPORT !!
    Only for unsupervised challenges.
    Please do not read in during the initialization phase.
    """

    @classmethod
    def from_challenge_prompt(cls, raw_prompt: str):
        return cls(raw_prompt=raw_prompt)

    @classmethod
    def from_serialized(cls, fields: Dict[str, List[str]]):
        positive_labels = []
        negative_labels = []
        for kb, labels in fields.items():
            kb = kb.lower()
            if "pos" in kb or kb.startswith("t"):
                positive_labels = labels
            elif "neg" in kb or kb.startswith("f"):
                negative_labels = labels
        return cls(positive_labels=positive_labels, negative_labels=negative_labels)

    @classmethod
    def from_binary_labels(cls, positive_labels: List[str], negative_labels: List[str]):
        return cls(positive_labels=positive_labels, negative_labels=negative_labels)
