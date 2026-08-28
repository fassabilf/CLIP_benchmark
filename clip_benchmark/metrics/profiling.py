"""Efficiency profiling: parameters, FLOPs and latency, split per tower.

Design notes
------------
Everything is derived from the objects `load_clip` already returns, so the
numbers describe the exact pipeline the accuracy numbers came from:

  * input resolution  → from `transform` (run it on a dummy PIL image)
  * context length    → from `tokenizer` (run it on a dummy caption)

Both towers are profiled through the public `encode_image` / `encode_text`
methods rather than through submodules. open_clip's plain `CLIP` has no
`.text` attribute (only `CustomTextCLIP` does) and the HF wrapper has neither,
so the methods are the only handle common to all three model loaders.

FLOPs are counted with `torch.utils.flop_counter.FlopCounterMode`, which
reports *true FLOPs* (a matmul MAC counts as 2). Papers that report "FLOPs"
via fvcore are usually reporting MACs, i.e. half of these numbers — the
`flops_convention` field in the output records which convention is in use.

Counting runs in fp32 without autocast (FLOPs are a property of the graph, not
of precision). Count on CUDA: on CPU the fused SDPA kernel is opaque to
`FlopCounterMode`, so the attention QK^T/AV matmuls go uncounted and every
image tower comes out ~0.36 GFLOPs light (ViT-T-16: 2.15 vs 2.51). Report the
CUDA counts; use the CPU run for latency only. Latency runs under the same `torch.autocast(enabled=amp)` the
eval metrics use, since that is what actually executes at eval time.
"""
import statistics
from contextlib import suppress

import torch
from PIL import Image
from torch.overrides import TorchFunctionMode
from torch.utils.flop_counter import FlopCounterMode

# Method names a CLIP-like model might expose for each tower, in priority order.
# Resolution stops at the first hit, so a model only needs one of them.
IMAGE_ENCODER_METHODS = ("encode_image", "get_image_features", "encode_visual",
                         "image_encoder", "visual_forward")
TEXT_ENCODER_METHODS = ("encode_text", "get_text_features", "encode_txt",
                        "text_encoder", "text_forward")

# A parameter belongs to the image tower if any component of its dotted name
# matches one of these. Covers open_clip (`visual.*`) and HF CLIP-likes
# (`model.vision_model.*`, `model.visual_projection.*`).
IMAGE_PARAM_HINTS = ("visual", "vision_model", "visual_projection", "image_encoder")
# Shared scalars that belong to neither tower.
OTHER_PARAM_LEAVES = ("logit_scale", "logit_bias", "logit_bias_")

DUMMY_CAPTION = "a photo of a dog"


# ─── input construction ──────────────────────────────────────────────────────

def example_image(transform, batch_size=1):
    """Run `transform` on a dummy PIL image to get the tensor the model sees."""
    img = Image.new("RGB", (512, 512), (128, 128, 128))
    t = transform(img)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"transform returned {type(t)}, expected a torch.Tensor")
    return t.unsqueeze(0).repeat(batch_size, *([1] * t.ndim))


def example_text(tokenizer, batch_size=1):
    """Tokenize a dummy caption `batch_size` times, exactly as the metrics do."""
    return tokenizer([DUMMY_CAPTION] * batch_size)


def to_device(x, device):
    """Move a tensor or a HF BatchEncoding/dict of tensors onto `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if hasattr(x, "to"):  # transformers.BatchEncoding
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def _text_shape(tokens):
    """[batch, context_length] of a tokenizer output, whatever its container."""
    if isinstance(tokens, torch.Tensor):
        return list(tokens.shape)
    ids = tokens["input_ids"] if "input_ids" in tokens else next(iter(tokens.values()))
    return list(ids.shape)


# ─── parameters ──────────────────────────────────────────────────────────────

def resolve_encoders(model, image_fn=None, text_fn=None):
    """Find each tower's entry point on an arbitrary CLIP-like model.

    Pass `image_fn`/`text_fn` explicitly for models that expose neither of the
    conventional method names.
    """
    def _find(explicit, candidates, which):
        if explicit is not None:
            return explicit
        for name in candidates:
            fn = getattr(model, name, None)
            if callable(fn):
                return fn
        raise RuntimeError(
            f"cannot find the {which} encoder on {type(model).__name__}: tried "
            f"{list(candidates)}. Pass {which}_fn= explicitly."
        )
    return (_find(image_fn, IMAGE_ENCODER_METHODS, "image"),
            _find(text_fn, TEXT_ENCODER_METHODS, "text"))


class _ParamTracker(TorchFunctionMode):
    """Record which of `param_ids` are consumed by ops inside the context."""

    def __init__(self, param_ids):
        super().__init__()
        self.param_ids = param_ids
        self.seen = set()

    def _scan(self, obj):
        if isinstance(obj, torch.Tensor):
            if id(obj) in self.param_ids:
                self.seen.add(id(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._scan(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._scan(item)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self._scan(args)
        self._scan(kwargs)
        return func(*args, **kwargs)


def attribute_params_by_execution(model, image_fn, text_fn, image_input, text_input):
    """Attribute parameters to towers by which ones each forward pass touches.

    Fully architecture-agnostic: no naming convention required, and raw
    `nn.Parameter`s (open_clip's `text_projection`, `positional_embedding`, …)
    are caught alongside module weights because tensor identity is what is
    tracked, not module structure.

    Must run outside autocast — under autocast the op receives a *cast copy* of
    the parameter, so identity matching would silently find nothing.
    """
    named = dict(model.named_parameters())
    ids = {id(p): n for n, p in named.items()}

    def _touched(fn, arg):
        tracker = _ParamTracker(set(ids))
        with tracker, torch.no_grad():
            fn(arg)
        return {ids[i] for i in tracker.seen}

    return _touched(image_fn, image_input), _touched(text_fn, text_input)


def is_projection(name):
    """True for the tower's final embedding projection head.

    Matched by an explicit `*projection*` name component (HF `visual_projection`,
    open_clip `text_projection`) or open_clip's bare `visual.proj` Parameter.
    Deliberately does *not* match `in_proj_weight` / `out_proj` / `mlp.c_proj`,
    which are internal attention/MLP weights, not the embedding head.
    """
    parts = name.split(".")
    if any("projection" in part for part in parts):
        return True
    return name in ("visual.proj", "visual.head.proj")


def vocab_embedding_params(model, min_rows=1000):
    """Names of the vocabulary look-up tables (`nn.Embedding` over a real vocab).

    TinyCLIP's headline parameter counts exclude these: "we do not count the
    number of parameters in the text embedding layer. It is a look-up table
    whose parameter size is the same as the models with the same hidden
    dimension and vocabulary size" (Wu et al., ICCV 2023, §4.1) — which is why
    the same tower is quoted as 3M there and measures 15.17M here.

    `min_rows` separates a vocabulary table from a positional one (77 or 577
    rows), so no naming convention is assumed.
    """
    names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding) and mod.num_embeddings >= min_rows:
            names.add(f"{mod_name}.weight" if mod_name else "weight")
    # open_clip's CLIP declares token_embedding as an nn.Embedding too, so the
    # module scan covers both loaders; a raw Parameter table would need a hint.
    return names


def split_params(model, image_names=None, text_names=None, extra_image_hints=()):
    """Split parameter counts into image tower / text tower / shared scalars.

    Two attribution modes:

    * ``image_names``/``text_names`` given — the sets measured by
      `attribute_params_by_execution`. Architecture-agnostic; preferred.
    * neither given — fall back to matching names against `IMAGE_PARAM_HINTS`.
      For open_clip this reproduces the published convention exactly
      (ViT-T-16 → 5.62M visual, 40.49M rest).

    `image` and `text` INCLUDE each tower's final projection head, since those
    weights execute at inference and contribute FLOPs. The heads are also
    reported separately as `image_projection` / `text_projection` because
    published numbers are inconsistent about them — e.g. TinyCLIP is commonly
    quoted as 8.15M/15.04M, which is `vision_model`/`text_model` with the two
    131K projections excluded. Subtract them to recover that convention.
    """
    hints = tuple(IMAGE_PARAM_HINTS) + tuple(extra_image_hints)
    by_execution = image_names is not None and text_names is not None
    vocab_names = vocab_embedding_params(model)

    image = text = other = unattributed = 0
    image_proj = text_proj = 0
    image_vocab = text_vocab = 0
    for name, p in model.named_parameters():
        parts = name.split(".")
        n = p.numel()
        if parts[-1] in OTHER_PARAM_LEAVES:
            other += n
            continue
        if by_execution:
            in_image, in_text = name in image_names, name in text_names
            if in_image and not in_text:
                bucket = "image"
            elif in_text and not in_image:
                bucket = "text"
            elif in_image and in_text:
                bucket = "other"   # genuinely shared between towers
            else:
                bucket = None      # never executed by either pass
        else:
            bucket = "image" if any(part in hints for part in parts) else "text"

        if bucket == "image":
            image += n
            if is_projection(name):
                image_proj += n
            if name in vocab_names:
                image_vocab += n
        elif bucket == "text":
            text += n
            if is_projection(name):
                text_proj += n
            if name in vocab_names:
                text_vocab += n
        elif bucket == "other":
            other += n
        else:
            unattributed += n

    total = sum(p.numel() for p in model.parameters())
    assert image + text + other + unattributed == total, "parameter split lost parameters"
    if image == 0:
        raise RuntimeError(
            f"no image-tower parameters found for {type(model).__name__}; pass "
            "extra_image_hints= or check that the image encoder resolved correctly"
        )
    return {
        "image": image, "text": text, "other": other, "total": total,
        "image_projection": image_proj, "text_projection": text_proj,
        "image_vocab_embedding": image_vocab, "text_vocab_embedding": text_vocab,
        "unattributed": unattributed,
        "method": "execution" if by_execution else "name",
    }


# ─── FLOPs ───────────────────────────────────────────────────────────────────

def count_flops(fn, arg):
    """True FLOPs of a single `fn(arg)` call, counted in fp32 with no autocast."""
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            fn(arg)
    return counter.get_total_flops()


# ─── latency ─────────────────────────────────────────────────────────────────

def measure_latency(fn, arg, device, warmup=20, runs=100, amp=True, memory=True):
    """Wall-clock latency of `fn(arg)`, in milliseconds.

    CUDA timings use cuda events and synchronize around every measured call, so
    what is timed is kernel completion rather than kernel launch. Returns
    mean/std/median/p90/min so tail behaviour stays visible.

    On CUDA, `memory=True` also reports the peak memory the measured passes
    reach (`peak_mem_alloc_mb` = tensor bytes, `peak_mem_reserved_mb` = what the
    caching allocator held). The peak counters are reset after the warmup so the
    number describes a steady-state forward pass, and it covers activations for
    THIS tower at THIS batch size on top of the whole model's resident weights.
    """
    is_cuda = torch.device(device).type == "cuda"
    autocast = torch.autocast(torch.device(device).type, enabled=amp)

    def _once():
        with torch.no_grad(), autocast:
            fn(arg)

    for _ in range(warmup):
        _once()
    if is_cuda:
        torch.cuda.synchronize()
        if memory:
            torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(runs):
        if is_cuda:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            start.record()
            _once()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            import time
            t0 = time.perf_counter()
            _once()
            times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms.sort()
    out = {
        "mean_ms": statistics.fmean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p50_ms": statistics.median(times_ms),
        "p90_ms": times_ms[min(int(0.9 * len(times_ms)), len(times_ms) - 1)],
        "min_ms": times_ms[0],
        "runs": runs,
        "warmup": warmup,
    }
    if is_cuda and memory:
        out["peak_mem_alloc_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        out["peak_mem_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    return out


# ─── driver ──────────────────────────────────────────────────────────────────

def profile(model, transform, tokenizer, device="cuda", amp=True,
            latency_batch_sizes=(1,), warmup=20, runs=100, latency=True,
            verbose=True, image_fn=None, text_fn=None, param_split="auto",
            extra_image_hints=(), memory=True):
    """Profile any CLIP-like model. Returns a JSON-serialisable dict.

    Model-agnostic: the two towers are found via `resolve_encoders` (or passed
    in as `image_fn`/`text_fn`), input shapes come from the model's own
    `transform`/`tokenizer`, and parameters are attributed by tracing which ones
    each forward pass actually touches.

    param_split
        ``"auto"``      execution tracing, falling back to name matching if the
                        trace attributes nothing (default)
        ``"execution"`` tracing only, error if it fails
        ``"name"``      name matching only

    FLOPs are always reported per sample (counted at batch size 1, fp32, no
    autocast — FLOPs are a property of the graph, not of precision). Latency is
    reported per tower per batch size, under the same autocast the eval metrics
    use.
    """
    model.eval()
    image_fn, text_fn = resolve_encoders(model, image_fn, text_fn)

    img1 = to_device(example_image(transform, 1), device)
    txt1 = to_device(example_text(tokenizer, 1), device)

    # ── parameters ──
    params = None
    if param_split in ("auto", "execution"):
        try:
            img_names, txt_names = attribute_params_by_execution(
                model, image_fn, text_fn, img1, txt1)
            params = split_params(model, img_names, txt_names,
                                  extra_image_hints=extra_image_hints)
        except Exception as e:
            if param_split == "execution":
                raise
            if verbose:
                print(f"  execution-traced param split failed ({type(e).__name__}: {e}); "
                      "falling back to name matching")
    if params is None:
        params = split_params(model, extra_image_hints=extra_image_hints)

    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "torch_version": torch.__version__,
        "precision": "amp" if amp else "fp32",
        "image_input_shape": list(img1.shape),
        "text_input_shape": _text_shape(txt1),
        "flops_convention": "torch.utils.flop_counter — true FLOPs (1 MAC = 2 FLOPs)",
        "params": params,
    }

    if verbose:
        print(f"  image input {result['image_input_shape']}  "
              f"text input {result['text_input_shape']}  "
              f"(param split: {params['method']})")

    # ── FLOPs ──
    image_flops = count_flops(image_fn, img1)
    text_flops = count_flops(text_fn, txt1)
    result["flops"] = {
        "image": image_flops,
        "text": text_flops,
        "total": image_flops + text_flops,
    }
    result["gflops"] = {k: v / 1e9 for k, v in result["flops"].items()}

    # ── latency ──
    if latency:
        result["latency"] = {}
        for bs in latency_batch_sizes:
            img = to_device(example_image(transform, bs), device)
            txt = to_device(example_text(tokenizer, bs), device)
            entry = {
                "image": measure_latency(image_fn, img, device, warmup, runs, amp, memory),
                "text": measure_latency(text_fn, txt, device, warmup, runs, amp, memory),
            }
            entry["total_mean_ms"] = entry["image"]["mean_ms"] + entry["text"]["mean_ms"]
            entry["images_per_sec"] = bs / (entry["image"]["mean_ms"] / 1e3)
            entry["texts_per_sec"] = bs / (entry["text"]["mean_ms"] / 1e3)
            result["latency"][str(bs)] = entry
            if verbose:
                mem = entry["image"].get("peak_mem_alloc_mb")
                mem_s = f"  peak {mem:6.0f} MB" if mem is not None else ""
                print(f"  bs={bs:<4d} image {entry['image']['p50_ms']:7.2f} ms  "
                      f"text {entry['text']['p50_ms']:7.2f} ms (median)  "
                      f"{entry['images_per_sec']:8.1f} img/s "
                      f"{entry['texts_per_sec']:8.1f} txt/s{mem_s}")

    return result


def assert_vocab_size(model, checkpoint_path, verbose=True):
    """Guard against the ViT-T-16 config ambiguity (CLIP-BPE 49408 vs SigLIP2 256000).

    The same arch name resolves to two different text configs depending on which
    open_clip checkout is on the path. Compare the built model's embedding
    against the checkpoint's, so a mismatched config fails loudly instead of
    silently reporting the wrong text-tower size.
    """
    emb = getattr(model, "token_embedding", None)
    if emb is None:
        with suppress(AttributeError):
            emb = model.text.token_embedding
    if emb is None:
        return None
    built = tuple(emb.weight.shape)

    state = torch.load(checkpoint_path, map_location="cpu")
    state = state.get("state_dict", state)
    key = next((k for k in state if k.endswith("token_embedding.weight")), None)
    if key is None:
        return None
    ckpt = tuple(state[key].shape)
    if built != ckpt:
        raise RuntimeError(
            f"config/checkpoint mismatch: built token_embedding {built} but "
            f"{checkpoint_path} has {ckpt}. The arch name resolves to a "
            f"different text config in this environment."
        )
    if verbose:
        print(f"  vocab check OK: token_embedding {built}")
    return built
