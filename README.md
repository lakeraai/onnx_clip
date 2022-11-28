# onnx_clip

## About
We replicate `OpenAI`'s [CLIP](https://github.com/openai/CLIP) without needing the
various `PyTorch` dependencies by heavily borrowing their code and adding some of our own. We do this by utilising a `.onnx` format of the model, a pure `NumPy` version of the tokenizer, 
and an accurate approximation of the [preprocess function.](https://github.com/openai/CLIP/blob/main/clip/clip.py#L79)
Due to this final approximation, the output logits do
not perfectly match those of `CLIP` but are close enough for our purposes.

## git lfs
This repository uses Git LFS for the `clip_model.onnx` file. Before cloning, run
```bash
git lfs install
```

## Installation
To install, run the following in the root of the repository:
```bash
pip install git+ssh://git@github.com/lakeraai/onnx_clip.git 
```

## Usage

All you need to do is call the `OnnxClip` model class. An example can be seen below.

```python
from onnx_clip import OnnxClip, softmax
from PIL import Image

image = Image.open("onnx_clip/data/CLIP.png").convert("RGB")
text = ["a photo of a man", "a photo of a woman"]
onnx_model = OnnxClip()
logits_per_image, logits_per_text = onnx_model.predict(image, text)
probas = softmax(logits_per_image)
```

