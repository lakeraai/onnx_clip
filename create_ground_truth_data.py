"""
Copyright 2023 Lakera AI. All Rights Reserved.

A simple script that loads predictions made by the original CLIP implementation
so that we can make sure that our version matches.

This is used to generate constants/fixtures for the tests.
"""
import os

import numpy as np
from PIL import Image
import clip  # https://github.com/openai/CLIP

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "onnx_clip/data/",
)

IMAGE_PATH = os.path.join(DATA_DIR, "franz-kafka.jpg")

TEXTS = [
    "a photo of a man",
    "a photo of a woman",
]


def main():
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    image = Image.open(IMAGE_PATH).convert("RGB")

    preprocessed_image = preprocess(image).unsqueeze(0)

    # image_tensor = torch.as_tensor(np.moveaxis(np.array(image), 2, 0))
    # preprocessed_image_2 = preprocess(image_tensor).unsqueeze(0)
    # assert torch.allclose(preprocessed_image, preprocessed_image_2)

    preprocessed_texts = clip.tokenize(TEXTS).to("cpu")

    image_embeddings = model.encode_image(preprocessed_image)
    text_embeddings = model.encode_text(preprocessed_texts)

    logits_per_image, logits_per_text = model(
        preprocessed_image, preprocessed_texts
    )

    image_embeddings = image_embeddings.detach().numpy()
    text_embeddings = text_embeddings.detach().numpy()

    print("expected_image_embeddings_sum =", float(np.sum(image_embeddings)))
    print(
        "expected_image_embeddings_part =",
        image_embeddings[0, :5].astype(float).tolist(),
    )
    print()
    print(
        "expected_text_embeddings_sums =",
        list(np.sum(text_embeddings, axis=1).astype(float)),
    )
    print(
        "expected_text_embeddings_part =",
        text_embeddings[0, :5].astype(float).tolist(),
    )
    print()
    print(
        "expected_logits_per_image =",
        logits_per_image.detach().numpy().astype(float).tolist(),
    )
    print()

    preprocessed_image_path = os.path.join(
        DATA_DIR, "expected_preprocessed_image.npy"
    )
    # np.save(preprocessed_image_path, preprocessed_image)
    print(f"Saved preprocessed image into {preprocessed_image_path}")


if __name__ == "__main__":
    main()
