# onnx_clip

An [ONNX](https://onnx.ai/)-based implementation of [CLIP](https://github.com/openai/CLIP) that doesn't
depend on `torch` or `torchvision`.

This works by
- running the text and vision encoders (the ViT-B/32 variant) in [ONNX Runtime](https://onnxruntime.ai/)
- using a pure NumPy version of the tokenizer
- using a pure NumPy+PIL version of the [preprocess function](https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L79).
  The PIL dependency could also be removed with minimal code changes - see `preprocessor.py`.

## git lfs
This repository uses Git LFS for the `clip_model.onnx` file. Make sure to do `git lfs install` before cloning.

In case you use the `onnx_clip` project not as a repo, but as a package, the model will be downloaded from
[the public S3 bucket](https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_model.onnx).

## Installation
To install, run the following in the root of the repository:
```bash
pip install .
```

## Usage

All you need to do is call the `OnnxClip` model class. An example can be seen below.

```python
from onnx_clip import OnnxClip, softmax
from PIL import Image

images = [Image.open("onnx_clip/data/franz-kafka.jpg").convert("RGB")]
text = ["a photo of a man", "a photo of a woman"]
onnx_model = OnnxClip()
logits_per_image, logits_per_text = onnx_model.predict(images, text)
probas = softmax(logits_per_image)
```

## Building & developing from source

**Note**: The following may give timeout errors due to the filesizes. If so, this can be fixed with poetry version 1.1.13 - see [this related issue.](https://github.com/python-poetry/poetry/issues/6009)

### Install, run, build and publish with Poetry

Install [Poetry](https://python-poetry.org/docs/)
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

To setup the project and create a virtual environment run the following command from the project's root directory.
```
poetry install
```

To build a source and wheel distribution of the library run the following command from the project's root directory.
```
poetry build
```

#### Instructions to publish the build artifacts for project maintainers
Copy this into your poetry config.toml file (or create a new one).
```
[repositories]
[repositories.onnx_clip]
url = "https://gitlab.com/api/v4/projects/41150990/packages/pypi"
```
The file should be located here on MacOs
```
~/Library/Application Support/pypoetry/config.toml
```
and here on Linux
```
~/.config/pypoetry/config.toml
```

With this setup you can now publish a package like so
```
poetry publish --repository onnx_clip -u <access_token_name> -p <access_token_key>
```
WARNING: Do not publish to the public pypi registry, e.g. always use the --repository option.
NOTE1: You must generate [an access token](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html)
with scope set to api.  
NOTE2: The push will fail if there is already a package with the same version. You can increment the version using [poetry](https://python-poetry.org/docs/cli/#version)
```
poetry version
```
or by manually changing the version number in pyproject.toml.

## Help

Please let us know how we can support: [earlyaccess@lakera.ai](mailto:earlyaccess@lakera.ai).

## LICENSE
See the [LICENSE](./LICENSE) file in this repository.

The `franz-kafka.jpg` is taken from [here](https://www.knesebeck-verlag.de/franz_kafka/p-1/270).