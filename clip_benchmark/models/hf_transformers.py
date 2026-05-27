"""HF-transformers dual-encoder loader (MetaCLIP-2, etc.).

Wraps an `AutoModel` + `AutoProcessor` into the (model, transform, tokenizer)
triple clip_benchmark expects, exposing `encode_image` / `encode_text` that
delegate to the HF `get_image_features` / `get_text_features` heads.
"""
import torch
from transformers import AutoModel, AutoProcessor


class HFCLIPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_image(self, pixel_values):
        out = self.model.get_image_features(pixel_values=pixel_values)
        return _unwrap(out)

    def encode_text(self, text_inputs):
        # BatchEncoding (HF tokenizer output) is a UserDict-like, not a real
        # dict — `isinstance(_, dict)` is False, but it does support `**`.
        if isinstance(text_inputs, torch.Tensor):
            out = self.model.get_text_features(input_ids=text_inputs)
        else:
            out = self.model.get_text_features(**text_inputs)
        return _unwrap(out)


def _unwrap(out):
    # MetaClip2Model (and some other HF CLIP-likes) returns
    # BaseModelOutputWithPooling whose `pooler_output` holds the projected
    # embedding; vanilla CLIPModel returns the tensor directly.
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    return out.pooler_output


def _make_transform(processor):
    image_processor = processor.image_processor

    def transform(img):
        out = image_processor(img, return_tensors="pt")
        return out["pixel_values"][0]

    return transform


def _make_tokenizer(processor, max_length):
    tok = processor.tokenizer

    def tokenize(texts):
        return tok(texts, padding="max_length", truncation=True,
                   max_length=max_length, return_tensors="pt")

    return tokenize


def _resolve_max_length(model):
    cfg = getattr(model, "config", None)
    tcfg = getattr(cfg, "text_config", None) if cfg is not None else None
    for src in (tcfg, cfg):
        n = getattr(src, "max_position_embeddings", None) if src else None
        if isinstance(n, int) and n > 0:
            return n
    return 77


def load_hf_transformers(model_name: str, pretrained: str = "",
                         cache_dir: str = None, device: str = "cpu"):
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    wrapper = HFCLIPWrapper(model).to(device).eval()
    max_length = _resolve_max_length(model)
    return wrapper, _make_transform(processor), _make_tokenizer(processor, max_length)
