import inspect
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import clip
import open_clip
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelCard:
    """
    Open-CLIP Benchmarks
    > https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

    Open-CLIP Model Profile
    > https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv

    """

    model_name: str
    """
    regular model name
    """

    tag: str = ""
    """
    pretrained
    """

    onnx_visual: str = ""
    onnx_textual: str = ""
    """
    Merged ONNX Model Name.
    If you decide to export the model on the huggingface, 
    please customize the onnx_visual and onnx_textual fields.
    """

    onnx_visual_path: Path = field(default_factory=Path)
    onnx_textual_path: Path = field(default_factory=Path)
    model_dir: Path = Path("model")
    """
    Default export path:
      - visual: [project_dir]/model/*[self.model_name]/[VERSION]/[self.onnx_visual]
      - textual: [project_dir]/model/*[self.model_name]/[VERSION]/[self.onnx_textual]
    """

    DEFAULT_TEXTUAL_FIELDS = {
        "export_params": True,
        "input_names": ["TEXT"],
        "output_names": ["TEXT_EMBEDDING"],
        "dynamic_axes": {"TEXT": {0: "text_batch_size"}, "TEXT_EMBEDDING": {0: "text_batch_size"}},
    }
    """
    Template parameter of the textual-part appended to `torch.onnx.export`
    """

    DEFAULT_VISUAL_FIELDS = {
        "export_params": True,
        "input_names": ["IMAGE"],
        "output_names": ["IMAGE_EMBEDDING"],
        "dynamic_axes": {
            "IMAGE": {0: "image_batch_size"},
            "IMAGE_EMBEDDING": {0: "image_batch_size"},
        },
    }
    """
    Template parameter of the visual-part appended to `torch.onnx.export`
    """

    def __post_init__(self):
        _prefix = self.model_name

        if not self.tag and (not self.onnx_visual or not self.onnx_textual):
            logging.warning(
                "If you decide to export the model on the huggingface, "
                "please customize the onnx_visual and onnx_textual fields."
            )
            _prefix = self.model_name.split(":")[-1]

        # fixme: more standardized model naming
        if not self.tag:
            inner_ = _prefix.split("/")[-1]
            self.onnx_visual = f"visual_CLIP_{inner_}.onnx"
            self.onnx_textual = f"textual_CLIP_{inner_}.onnx"
        else:
            self.onnx_visual = f"visual_CLIP_{_prefix}.{self.tag}.onnx"
            self.onnx_textual = f"textual_CLIP_{_prefix}.{self.tag}.onnx"

        _suffix = 1
        for _ in range(100):
            pre_dir = self.model_dir.joinpath(_prefix, f"v{_suffix}")
            if not pre_dir.exists():
                pre_dir.mkdir(parents=True, exist_ok=True)
                break
            _suffix += 1

        self.onnx_visual_path = self.model_dir.joinpath(_prefix, f"v{_suffix}", self.onnx_visual)
        self.onnx_textual_path = self.model_dir.joinpath(_prefix, f"v{_suffix}", self.onnx_textual)
        logging.info(self.onnx_textual_path)
        logging.info(self.onnx_textual_path)

    @classmethod
    def from_template(cls, template):
        return cls(
            **{
                key: (template[key] if val.default == val.empty else template.get(key, val.default))
                for key, val in inspect.signature(cls).parameters.items()
            }
        )

    def __call__(self, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_image_path = kwargs.get("dummy_image_path")

        if not self.tag:
            model, preprocess = open_clip.create_model_from_pretrained(
                self.model_name, device=device
            )
        else:
            model, preprocess = open_clip.create_model_from_pretrained(
                self.model_name, self.tag, device=device
            )

        self.to_onnx_visual(model, preprocess, device, dummy_image_path)
        self.to_onnx_textual(model, device)

        logging.info(f"Successfully exported model - path={self.model_dir}")

        return model, preprocess

    def to_onnx_visual(self, model, preprocess, device, dummy_image_path: Path = None):
        dummy_image_path = dummy_image_path or Path("franz-kafka.jpg")
        dummy_image = preprocess(Image.open(dummy_image_path)).unsqueeze(0).to(device)

        model.forward = model.encode_image
        torch.onnx.export(
            model=model,
            args=(dummy_image,),
            f=f"{self.onnx_visual_path}",
            **self.DEFAULT_VISUAL_FIELDS,
        )

    def to_onnx_textual(self, model, device):
        dummy_text = clip.tokenize(
            ["This is a photo of cat.", "This is a photo of dog.", "This is a photo of girl."]
        ).to(device)

        model.forward = model.encode_text
        torch.onnx.export(
            model=model,
            args=(dummy_text,),
            f=f"{self.onnx_textual_path}",
            **self.DEFAULT_TEXTUAL_FIELDS,
        )
