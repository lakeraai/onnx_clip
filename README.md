# lakera-clip

## About
The purpose of this repository is to replicate the functionality of [CLIP](https://github.com/openai/CLIP) without needing the
various `PyTorch` dependencies. We do this by utilising a `.onnx` format of the model, a pure `NumPy` version of the tokenizer, 
and an accurate approximation of the `preprocess` function. Due to this final approximation, the output logits do
not perfectly match those of `CLIP` but are close enough for our purposes.

## git lfs
This repository uses Git LFS for the `clip_model.onnx` file. Make sure to do `git lfs install` before cloning.

## Installation
To install, run the following in the root of the repository:
`pip install -e .`


## Example usage

```python
from onnx_clip import OnnxClip
from PIL import Image

image = Image.open("onnx_clip/data/CLIP.png")
text = ["a photo of a man", "a photo of a woman"]
onnx_model = OnnxClip()
logits_per_image, logits_per_text = onnx_model.run(image, text)
```

