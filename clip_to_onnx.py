"""
Load and save the CLIP model in ONNX format such that pytorch is not required.
"""
import os

import clip
import torch
from PIL import Image

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "onnx_clip/data/",
)

CLIP_IMAGE_ONNX_EXPORT_PATH = "clip_image_model_vitb32.onnx"
CLIP_TEXT_ONNX_EXPORT_PATH = "clip_text_model_vitb32.onnx"


def generate_dummy_text(device):
    """
    In order to create the ONNX model, the model is actually run once. In our case,
    this means running clip_model(image, text). We therefore require some text and image so that the
    model can be run. Once this is done, the model architecture is correctly preserved and exported.
    The text for this method is called dummy text. It does not have to mean anything, but must be in the correct
    format such that the model can be run.
    Args:
        device: current device
    Returns:
        pytorch clip-tokenized version of text.
        This text can be anything, it just needs to be in a format that can be fed
        to the model.
    """
    return clip.tokenize(
        [
            "a photo taken during the day",
            "a photo taken at night",
            "a photo taken of Mickey Mouse",
        ]
    ).to(device)


def generate_dummy_image(preprocess, device):
    """
    As above with the dummy text, we require an image to be fed to the model such that the CLIP model can be run
    once. Changing this image does not create a different model, it is simply a placeholder.
    Args:
        preprocess: clip preprocess function
        device: current device

    Returns:
        numpy embeddings of an image.
        This image can be anything, it just needs to be in a format that can be fed to the model, like the text.

    """
    dummy_image_path = os.path.join(DATA_DIR, "franz-kafka.jpg")
    return preprocess(Image.open(dummy_image_path)).unsqueeze(0).to(device)


def export_onnx(
    model,
    inputs,
    input_names,
    output_names,
    export_path,
    dynamic_axes=None,
) -> None:
    """
    Saves a model in .onnx format.

    Args:
        model: the model to export
        inputs: the inputs required for the model to work. In our case, a tuple of format (dummy image, dummy text)
        input_names: the names of the inputs
        output_names: the names of the outputs
        export_path: where to save the model
        dynamic_axes: names of the inputs and batches.

    """
    export_path = os.path.join(DATA_DIR, export_path)
    torch.onnx.export(
        model=model,
        args=inputs,
        f=export_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        # This is the lowest opset version that still works. There's a warning
        # about "Exporting aten::index operator of advanced indexing" but it's
        # emitted for every opset up to 16, the highest version supported by
        # torch.onnx.export().
        opset_version=9,
        dynamic_axes=dynamic_axes,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    dummy_text = generate_dummy_text(device)
    dummy_image = generate_dummy_image(preprocess, device)

    # As a hack, we replace the forward() function, which the
    # `torch.onnx.export` function uses as the entrypoint to the model.
    # This allows us to separately export the image and text encoders.
    # These don't share weights, so this is not wasteful.
    model.forward = model.encode_image
    export_onnx(
        model=model,
        inputs=(dummy_image,),
        input_names=["IMAGE"],
        output_names=["IMAGE_EMBEDDING"],
        dynamic_axes={
            "IMAGE": {
                0: "image_batch_size",
            },
            "IMAGE_EMBEDDING": {
                0: "image_batch_size",
            },
        },
        export_path=CLIP_IMAGE_ONNX_EXPORT_PATH,
    )

    model.forward = model.encode_text
    export_onnx(
        model=model,
        inputs=(dummy_text,),
        input_names=["TEXT"],
        output_names=["TEXT_EMBEDDING"],
        dynamic_axes={
            "TEXT": {
                0: "text_batch_size",
            },
            "TEXT_EMBEDDING": {
                0: "text_batch_size",
            },
        },
        export_path=CLIP_TEXT_ONNX_EXPORT_PATH,
    )


if __name__ == "__main__":
    main()
