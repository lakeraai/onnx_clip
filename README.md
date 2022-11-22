# lakera-clip
## git lfs
This repository uses Git LFS for the `clip_model.onnx` file. Make sure to do `git lfs install` before cloning.

## Installation
To install, run the following in the root of the repository:
`pip install -e.`


## Example usage

```python
from lakera_clip import Model
from PIL import Image

image = Image.open("lakera_clip/data/CLIP.png")
text = ["a photo of a man", "a photo of a woman"]
lakera_model = Model()
logits_per_image, logits_per_text = lakera_model.run(image, text)
```

