import os

import numpy as np
import pytest
from PIL import Image

from onnx_clip import OnnxClip, softmax, get_similarity_scores

IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../onnx_clip/data/franz-kafka.jpg",
)

ALLOWED_MODELS = ["ViT-B/32", "RN50"]

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

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_bad_image_input(model):
    """
    Test that a non-PIL input is bad for an image.
    """
    _, texts = load_image_and_texts()

    onnx_model = OnnxClip(model=model)
    with pytest.raises(TypeError):
        onnx_model.get_image_embeddings("bad image input")

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_bad_image_channels(model):
    """
    Test that a 4-channel image raises the appropriate error.
    """
    image, texts = load_image_and_texts(to_rgba=True)

    onnx_model = OnnxClip(model=model)
    with pytest.raises(ValueError):
        onnx_model.get_image_embeddings([image])

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_bad_text_input(model):
    """
    Test that a non-tokenized input text is bad for model.
    """
    image, _ = load_image_and_texts()

    onnx_model = OnnxClip(model=model)
    with pytest.raises(TypeError):
        onnx_model.get_text_embeddings([image])

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_embedding_size_redundancy(model):
    """
    Test that calling EMBEDDING_SIZE returns a 
    no-longer-supported error.
    """
    onnx_model = OnnxClip(model=model)
    with pytest.raises(RuntimeError):
        onnx_model.EMBEDDING_SIZE

def test_softmax_values():
    """
    Test the softmax function works as expected.
    """

    logits = np.array([[0, 10, -10]])
    assert all(np.isclose(np.sum(softmax(logits), axis=1), 1))

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_image_model_runs(model):
    image, _ = load_image_and_texts()
    n_images = 3

    onnx_model = OnnxClip(model=model)
    embeddings = onnx_model.get_image_embeddings(n_images * [image])

    assert embeddings.shape == (n_images, onnx_model.embedding_size)

    # See create_ground_truth_data.py
    if model == "ViT-B/32":
        expected_image_embeddings_sum = 7.557152271270752
        expected_image_embeddings_part = [
            -0.07944782078266144,
            0.22216692566871643,
            -0.04425959661602974,
            -0.1021476462483406,
            0.04593294858932495,
        ]
    elif model == "RN50":
        expected_image_embeddings_part = [
            -0.05500680208206177, 
            0.017263656482100487, 
            0.004518476314842701, 
            -0.0013883080100640655, 
            0.01354439090937376,
        ]
        expected_image_embeddings_sum = -1.4342610836029053

    assert np.allclose(
        embeddings[0, :5], expected_image_embeddings_part, atol=1e-6, rtol=1e-4
    )
    assert np.isclose(embeddings[0].sum(), expected_image_embeddings_sum)

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_text_model_runs(model):
    _, texts = load_image_and_texts()

    onnx_model = OnnxClip(model=model)
    embeddings = onnx_model.get_text_embeddings(texts)

    assert embeddings.shape == (len(texts), onnx_model.embedding_size)

    # See create_ground_truth_data.py
    if model == "ViT-B/32":
        expected_text_embeddings_sums = [9.667448043823242, 10.100772857666016]
        expected_text_embeddings_part = [
            -0.26425981521606445,
            0.3245568573474884,
            -0.022752312943339348,
            0.20319020748138428,
            -0.00989810936152935,
        ]
    elif model == "RN50":
        expected_text_embeddings_sums = [-23.376190185546875, -21.070472717285156]
        expected_text_embeddings_part = [
            0.3766278028488159, 
            0.21154774725437164, 
            0.08376261591911316, 
            -0.0009389406186528504, 
            0.28522634506225586,
        ]

    assert np.allclose(
        embeddings[0, :5], expected_text_embeddings_part, atol=1e-5
    )
    assert np.allclose(embeddings.sum(axis=1), expected_text_embeddings_sums)

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_model_runs(model):
    """
    Test full process.
    """
    image, texts = load_image_and_texts()
    n_images = 3
    n_text = 2

    onnx_model = OnnxClip(model=model)

    image_embeddings = onnx_model.get_image_embeddings([image] * n_images)
    text_embeddings = onnx_model.get_text_embeddings(texts)

    logits_per_image = get_similarity_scores(image_embeddings, text_embeddings)

    assert logits_per_image.shape == (n_images, n_text)

    # See create_ground_truth_data.py.
    if model == "ViT-B/32":
        expected_logits_per_image = [[27.878917694091797, 23.396446228027344]]
        expected_probabilities = [[0.9888209104537964, 0.011179053224623203]]
    elif model == "RN50":
        expected_logits_per_image = [[20.8167724609375, 17.331161499023438]]
        expected_probabilities = [[0.9702755808830261, 0.029724428430199623]]
    assert np.allclose(logits_per_image[:1], expected_logits_per_image)

    probas = softmax(logits_per_image)
    assert probas.shape == (n_images, n_text)
    assert np.allclose(probas[:1], expected_probabilities, atol=1e-6)

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_batching(model):
    """Check that using batching preserves the expected output."""
    image, (text, *_) = load_image_and_texts()

    onnx_model = OnnxClip(model=model, batch_size=2)

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
        assert batched_embeddings.shape == (n_items, onnx_model.embedding_size)
        for embedding in batched_embeddings:
            assert np.array_equal(embedding, correct_embedding)

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_iterator(model):
    image, (text, *_) = load_image_and_texts()

    n_items = 5
    image_iterator = iter([image] * n_items)
    text_iterator = iter([text] * n_items)

    onnx_model = OnnxClip(model=model, batch_size=2)

    image_embeddings = onnx_model.get_image_embeddings(image_iterator)
    text_embeddings = onnx_model.get_text_embeddings(text_iterator)

    assert image_embeddings.shape == (n_items, onnx_model.embedding_size)
    assert text_embeddings.shape == (n_items, onnx_model.embedding_size)

    # We've gone through the whole iterator
    with pytest.raises(StopIteration):
        next(image_iterator)

    with pytest.raises(StopIteration):
        next(text_iterator)

@pytest.mark.parametrize("model", ALLOWED_MODELS)
def test_empty(model):
    """Handle empty text and image inputs, both with and without batching"""

    onnx_model = OnnxClip(model=model, batch_size=2)
    assert onnx_model.get_image_embeddings([]).shape == (0, onnx_model.embedding_size)
    assert onnx_model.get_text_embeddings([]).shape == (0, onnx_model.embedding_size)

    onnx_model = OnnxClip(model=model)
    assert onnx_model.get_image_embeddings([]).shape == (0, onnx_model.embedding_size)
    assert onnx_model.get_text_embeddings([]).shape == (0, onnx_model.embedding_size)


def test_get_similarity_scores_broadcasting():
    embeddings = np.eye(3, 512)  # Rectangular identity matrix

    scores = get_similarity_scores(embeddings, embeddings)
    assert scores.shape == (3, 3)
    assert np.allclose(scores, np.eye(3, 3) * 100)

    scores = get_similarity_scores(embeddings[0], embeddings)
    assert scores.shape == (3,)
    assert np.allclose(scores, np.array([100, 0, 0]))

    scores = get_similarity_scores(embeddings, embeddings[0])
    assert scores.shape == (3,)
    assert np.allclose(scores, np.array([100, 0, 0]))

    scores = get_similarity_scores(embeddings[0], embeddings[0])
    assert scores.shape == ()
    assert isinstance(scores, float)  # NumPy float counts here too
    assert np.allclose(scores, 100)
