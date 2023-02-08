import os

import numpy as np
import pytest
from PIL import Image

from onnx_clip import OnnxClip, softmax, get_similarity_scores

IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../onnx_clip/data/franz-kafka.jpg",
)


def load_image_and_texts(to_rgba=False):
    """
    Load a test image and two text. Convert the image to RGB or RGBA.

    Returns:
        (test_image, test_text)
    """
    image = Image.open(IMAGE_PATH).convert("RGBA" if to_rgba else "RGB")
    texts = [
        "a photo of a man",
        "a photo of a woman",
    ]
    return image, texts


def test_bad_image_input():
    """
    Test that a non-PIL input is bad for an image.
    """
    _, texts = load_image_and_texts()

    onnx_model = OnnxClip()
    with pytest.raises(TypeError):
        onnx_model.get_image_embeddings("bad image input")


def test_bad_image_channels():
    """
    Test that a 4-channel image raises the appropriate error.
    """
    image, texts = load_image_and_texts(to_rgba=True)

    onnx_model = OnnxClip()
    with pytest.raises(ValueError):
        onnx_model.get_image_embeddings([image])


def test_bad_text_input():
    """
    Test that a non-tokenized input text is bad for model.
    """
    image, _ = load_image_and_texts()

    onnx_model = OnnxClip()
    with pytest.raises(TypeError):
        onnx_model.get_text_embeddings([image])


def test_softmax_values():
    """
    Test the softmax function works as expected.
    """

    logits = np.array([[0, 10, -10]])
    assert all(np.isclose(np.sum(softmax(logits), axis=1), 1))


def test_image_model_runs():
    image, _ = load_image_and_texts()
    n_images = 3

    onnx_model = OnnxClip()
    embeddings = onnx_model.get_image_embeddings(n_images * [image])

    assert embeddings.shape == (n_images, OnnxClip.EMBEDDING_SIZE)

    # See create_ground_truth_data.py
    expected_image_embeddings_sum = 7.557152271270752
    expected_image_embeddings_part = [
        -0.07944782078266144,
        0.22216692566871643,
        -0.04425959661602974,
        -0.1021476462483406,
        0.04593294858932495,
    ]

    assert np.allclose(
        embeddings[0, :5], expected_image_embeddings_part, atol=1e-6, rtol=1e-4
    )
    assert np.isclose(embeddings[0].sum(), expected_image_embeddings_sum)


def test_text_model_runs():
    _, texts = load_image_and_texts()

    onnx_model = OnnxClip()
    embeddings = onnx_model.get_text_embeddings(texts)

    assert embeddings.shape == (len(texts), OnnxClip.EMBEDDING_SIZE)

    # See create_ground_truth_data.py
    expected_text_embeddings_sums = [9.667448043823242, 10.100772857666016]
    expected_text_embeddings_part = [
        -0.26425981521606445,
        0.3245568573474884,
        -0.022752312943339348,
        0.20319020748138428,
        -0.00989810936152935,
    ]

    assert np.allclose(
        embeddings[0, :5], expected_text_embeddings_part, atol=1e-5
    )
    assert np.allclose(embeddings.sum(axis=1), expected_text_embeddings_sums)


def test_model_runs():
    """
    Test full process.
    """
    image, texts = load_image_and_texts()
    n_images = 3
    n_text = 2

    onnx_model = OnnxClip()

    image_embeddings = onnx_model.get_image_embeddings([image] * n_images)
    text_embeddings = onnx_model.get_text_embeddings(texts)

    logits_per_image = get_similarity_scores(image_embeddings, text_embeddings)

    assert logits_per_image.shape == (n_images, n_text)

    # See create_ground_truth_data.py.
    expected_logits_per_image = [[27.878917694091797, 23.396446228027344]]
    assert np.allclose(logits_per_image[:1], expected_logits_per_image)

    probas = softmax(logits_per_image)
    assert probas.shape == (n_images, n_text)

    # See create_ground_truth_data.py
    expected_probabilities = [[0.9888209104537964, 0.011179053224623203]]
    assert np.allclose(probas[:1], expected_probabilities, atol=1e-6)


def test_batching():
    """Check that using batching preserves the expected output."""
    image, (text, *_) = load_image_and_texts()

    onnx_model = OnnxClip(batch_size=2)

    image_embedding = onnx_model.get_image_embeddings([image])[0]
    text_embedding = onnx_model.get_text_embeddings([text])[0]

    n_items = 5
    batched_image_embeddings = onnx_model.get_image_embeddings(
        [image] * n_items
    )
    batched_text_embeddings = onnx_model.get_text_embeddings([text] * n_items)

    for batched_embeddings, correct_embedding in [
        (batched_image_embeddings, image_embedding),
        (batched_text_embeddings, text_embedding),
    ]:
        assert batched_embeddings.shape == (n_items, OnnxClip.EMBEDDING_SIZE)
        for embedding in batched_embeddings:
            assert np.array_equal(embedding, correct_embedding)


def test_iterator():
    image, (text, *_) = load_image_and_texts()

    n_items = 5
    image_iterator = iter([image] * n_items)
    text_iterator = iter([text] * n_items)

    onnx_model = OnnxClip(batch_size=2)

    image_embeddings = onnx_model.get_image_embeddings(image_iterator)
    text_embeddings = onnx_model.get_text_embeddings(text_iterator)

    assert image_embeddings.shape == (n_items, OnnxClip.EMBEDDING_SIZE)
    assert text_embeddings.shape == (n_items, OnnxClip.EMBEDDING_SIZE)

    # We've gone through the whole iterator
    with pytest.raises(StopIteration):
        next(image_iterator)

    with pytest.raises(StopIteration):
        next(text_iterator)
