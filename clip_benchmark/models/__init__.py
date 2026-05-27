from typing import Union
import torch
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip
from .hf_transformers import load_hf_transformers

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip,
    "hf_transformers": load_hf_transformers,
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def auto_model_type(model_name: str) -> str:
    """Infer model_type from the model identifier.

    - ``hf-hub:<repo>``  → open_clip (open_clip ships its own HF-hub loader)
    - ``<org>/<name>``   → hf_transformers (vanilla HF transformers repo)
    - anything else      → open_clip (arch name like ``ViT-B-32``)
    """
    if model_name.startswith("hf-hub:"):
        return "open_clip"
    if "/" in model_name:
        return "hf_transformers"
    return "open_clip"


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
):
    if model_type == "auto":
        model_type = auto_model_type(model_name)
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
