from pathlib import Path

from clip2onnx.model_card import ModelCard
from clip2onnx.templates import ViT_B_32, HF_ViT_L_14

project_dir = Path(__file__).parent.parent
assets_dir = project_dir.joinpath("assets")
dummy_image_path = assets_dir.joinpath("hello-world.jpg")
model_dir = project_dir.joinpath("model")


def demo():
    for template in [HF_ViT_L_14, ViT_B_32]:
        template.update({"model_dir": model_dir})
        model_card = ModelCard.from_template(template)
        model_card(dummy_image_path=dummy_image_path)


if __name__ == "__main__":
    demo()
