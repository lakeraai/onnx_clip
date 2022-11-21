# lakera-clip

## Example usage

```python
from lakera_clip import model, Tokenizer, preprocess
from PIL import Image

image = Image.open("lakera_clip/data/CLIP.png")
text = ["a photo of a man", "a photo of a woman"]

lakera_preprocess = preprocess()
lakera_tokenizer = Tokenizer()
lakera_model = model()

image_embeddings = lakera_preprocess.encode_image(image)
text_embeddings = lakera_tokenizer.encode_text(text)

logits_per_image, logits_per_text = lakera_model.run(image_embeddings, text_embeddings)
```

