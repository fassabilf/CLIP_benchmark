# REPORT — metaclip2_kd: ViT-T-16 student distilled from MetaCLIP2-B16-worldwide

**Run:** `multilingual-clip-kd/.../metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v1`
**Eval date:** 2026-05-31 · **Suite:** ImageNet-1k, Babel-ImageNet, XM3600, Flickr30k-200, XTD-200, CVQA
**Env:** `mc2_eval_env` pinned to **habibi's own open_clip** (copy at `open_clip_phabibi/src`, v3.2.0),
arch `ViT-T-16` = **CLIP-BPE** (vocab 49408, ctx 77 — matches checkpoint `token_embedding (49408,384)`).
**Predictions:** `fassabilf/sea-clip-eval-predictions` (HF, private), folders `mc2_e{0,8,16,24,32}/`.

## Setup notes
- Teacher = **MetaCLIP2-ViT-B-16-worldwide** (`metaclip2_b16.pt`), 3-source SEA blend
  (cultural-ground + WIT + bloom). The student text tower is **CLIP-BPE**, *not* the
  SigLIP2 256000 vocab of the earlier students — so eval used habibi's repo where
  `--model ViT-T-16` resolves to the CLIP-BPE config (open_clip_edit's `ViT-T-16` is the
  SigLIP2 config; using it would be wrong).
- **Smoke gate passed exactly:** ImageNet e8 acc1 = **0.03814** vs training-logged 0.038 →
  checkpoint/arch/tokenizer all correct, pipeline trusted.
- **`clipkd_vit_t_16_init_clean.pt` (= e0) is NOT a random init** — it is a fully-trained
  **English** CLIP-BPE student (ImageNet 0.426, mean-ENG 0.448 ≈ the S1 student). The run
  continued KD from this English init on SEA-only data.

## Results (image→text R@1 for retrieval, acc1 for classification)

| benchmark | grp | e0 (init) | e8 | e16 | e24 | e32 | **B16 teacher** | H14 (ceiling) | mv1_e32 (old) |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet1k | ENG | 0.426 | 0.038 | 0.047 | 0.056 | 0.056 | **0.711** | 0.813 | 0.005 |
| Babel-IN | ENG | 0.363 | 0.036 | 0.043 | 0.051 | 0.050 | 0.636 | 0.746 | 0.004 |
| Babel-IN | SEA | 0.042 | 0.054 | 0.077 | **0.086** | 0.084 | **0.424** | 0.549 | 0.012 |
| XM3600 | ENG | 0.394 | 0.027 | 0.024 | 0.024 | 0.024 | 0.460 | 0.477 | 0.003 |
| XM3600 | SEA | 0.007 | 0.024 | 0.039 | 0.045 | **0.046** | **0.526** | 0.613 | 0.003 |
| Flickr30k-200 | ENG | 0.599 | 0.024 | 0.032 | 0.022 | 0.026 | 0.761 | 0.839 | 0.002 |
| Flickr30k-200 | SEA | 0.011 | 0.013 | 0.021 | 0.024 | **0.025** | 0.450 | 0.596 | 0.003 |
| XTD-200 | ENG | 0.510 | 0.033 | 0.044 | 0.042 | 0.044 | 0.637 | 0.716 | 0.006 |
| XTD-200 | SEA | 0.017 | 0.025 | 0.032 | **0.039** | 0.037 | 0.381 | 0.512 | 0.006 |
| CVQA | EN | 0.395 | 0.284 | 0.281 | 0.276 | 0.271 | 0.538 | 0.614 | 0.254 |
| CVQA | LOCAL | 0.263 | 0.264 | 0.264 | 0.269 | 0.264 | 0.504 | 0.574 | 0.256 |
| **MEAN** | **ENG** | **0.448** | 0.074 | 0.079 | 0.079 | 0.079 | **0.624** | 0.701 | 0.046 |
| **MEAN** | **SEA** | 0.068 | 0.076 | 0.087 | **0.093** | 0.091 | **0.457** | 0.569 | 0.056 |

## Findings

1. **Catastrophic forgetting of English.** Starting from a strong English init (mean-ENG
   0.448), KD on SEA-only data collapsed English to ~0.079 — it traded away nearly all of
   the English ability the init had.
2. **SEA genuinely improves, unlike the old runs.** This is *not* the total
   representation-collapse of mv1/wit (where ENG **and** SEA sat at random). SEA moves up
   monotonically with KD: XM3600 SEA 0.007→0.046 (~6.5×), Babel SEA 0.042→0.086 (~2×).
   The multilingual teacher does transfer SEA signal — direction correct.
3. **But weak and far from the teacher.** Student SEA peaks at ~0.093 (mean) while the
   **B16 teacher it distilled from already does 0.457**. KD covered only ~20% of the way
   to the teacher on SEA. CVQA stays near random (LOCAL ~0.26).
4. **Plateau by ~e16–e24.** SEA gains flatten and e32 dips slightly vs e24 on several
   benchmarks (Babel SEA 0.086→0.084, XTD SEA 0.039→0.037).

## Per-script / per-language breakdown (mean across Babel+XM3600+Flickr+XTD)

Added as rows at the end of `benchmark_aggregate.csv` (`MEAN SEA-Latin`,
`MEAN SEA-nonLatin`, and `MEAN-LANG <lang>`). Script split: non-Latin = my (Myanmar),
th (Thai); Latin = id, jv, ms, su, vi.

| group | e0 | e8 | e16 | e24 | e32 | B16 teacher |
|---|---|---|---|---|---|---|
| **SEA-Latin** | 0.026 | 0.038 | 0.054 | 0.061 | **0.060** | 0.473 |
| **SEA-nonLatin** | 0.001 | 0.007 | 0.015 | 0.018 | **0.017** | 0.379 |
| en | 0.458 | 0.032 | 0.038 | 0.039 | 0.040 | 0.641 |
| id | 0.037 | 0.043 | 0.061 | 0.067 | **0.069** | 0.582 |
| ms | 0.031 | 0.044 | 0.057 | 0.069 | 0.067 | 0.528 |
| jv | 0.037 | 0.041 | 0.061 | 0.062 | 0.061 | 0.320 |
| vi | 0.008 | 0.029 | 0.051 | 0.057 | 0.056 | 0.540 |
| su | 0.038 | 0.041 | 0.046 | 0.055 | 0.052 | 0.258 |
| th | 0.002 | 0.010 | 0.018 | 0.024 | 0.022 | 0.450 |
| my | 0.001 | 0.004 | 0.008 | 0.007 | 0.008 | 0.251 |

**The student's bottleneck is its CLIP-BPE tokenizer.** Latin SEA reaches ~0.06 while
non-Latin sits at ~0.017 (Thai 0.022, Myanmar 0.008) — yet the B16 teacher (XLM-V
tokenizer) handles those scripts fine (th 0.450, my 0.251). So the multilingual signal
exists in the teacher; the byte-level CLIP-BPE student simply cannot represent Thai /
Myanmar script. Among Latin SEA, id/ms/vi gain most; vi has the largest lift (0.008→0.056)
since the English init covered Vietnamese diacritics poorly.

## Takeaway / next
The MetaCLIP2 teacher escapes the mv1 collapse and lifts SEA above the floor, but a
SEA-only KD from an English init yields severe forgetting + weak SEA acquisition. Likely
fixes: (a) mix English data back into the KD set to retain the init's strength, or
(b) init from scratch so there is no English to forget, isolating true SEA transfer, or
(c) **swap the student tokenizer to a multilingual one** (XLM-V / SigLIP2) — the
per-script result shows CLIP-BPE structurally caps non-Latin (Thai/Myanmar) regardless of
teacher quality.
