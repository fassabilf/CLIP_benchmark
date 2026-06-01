# REPORT — metaclip2_kd v2 (CC12M baseline): ViT-T-16 student distilled from MetaCLIP2-B16-worldwide on CC12M

**Run:** `multilingual-clip-kd/.../metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v2`
(SLURM job name `mclipkd-cc12m-baseline`)
**Eval date:** 2026-06-01 · **Suite:** ImageNet-1k, Babel-ImageNet, XM3600, Flickr30k-200, XTD-200, CVQA
**Env:** `mc2_eval_env` pinned to **habibi's own open_clip** (`open_clip_phabibi/src`, v3.2.0),
arch `ViT-T-16` = **CLIP-BPE** (vocab 49408, ctx 77).
**Predictions:** `fassabilf/sea-clip-eval-predictions` (HF, private), folders `mc2cc_e{8,16,24,32}/`.

## What this run is (vs the v1 run)

Same student / init / teacher / loss as the v1 run — the **only** difference is the training
data: **CC12M (10.97M English image–text pairs)** instead of the v1 3-source SEA blend
(cultural-ground + WIT + bloom). It is the **English baseline** for the SEA-transfer study.

- Teacher = **MetaCLIP2-ViT-B-16-worldwide** (`metaclip2_b16.pt`), loss `clipkd`, 32ep, lr 2e-3.
- Init = the shared `clipkd_vit_t_16_init_clean.pt` = a fully-trained **English** CLIP-BPE student
  (ImageNet 0.426, mean-ENG 0.448 ≈ S1). The e0 column below reuses the existing `mc2_e0` eval.
- **Smoke gate passed exactly:** ImageNet e32 acc1 = **0.40038** vs training-logged 0.4007 →
  arch/tokenizer/env all correct, pipeline trusted.

## Results (image→text R@1 for retrieval, acc1 for classification)

| benchmark | grp | e0 (init) | cc e8 | cc e16 | cc e24 | cc e32 | v1 SEA-blend e32 | **B16 teacher** |
|---|---|---|---|---|---|---|---|---|
| ImageNet1k | ENG | 0.426 | 0.296 | 0.333 | 0.381 | **0.400** | 0.056 | **0.711** |
| Babel-IN | ENG | 0.363 | 0.259 | 0.288 | 0.329 | **0.344** | 0.050 | 0.636 |
| Babel-IN | SEA | 0.042 | 0.038 | 0.041 | 0.040 | 0.040 | **0.084** | **0.424** |
| XM3600 | ENG | 0.394 | 0.268 | 0.300 | 0.331 | 0.326 | 0.024 | 0.460 |
| XM3600 | SEA | 0.007 | 0.006 | 0.008 | 0.007 | 0.007 | **0.046** | **0.526** |
| Flickr30k-200 | ENG | 0.599 | 0.354 | 0.381 | 0.414 | **0.419** | 0.026 | 0.761 |
| Flickr30k-200 | SEA | 0.011 | 0.009 | 0.007 | 0.010 | 0.007 | **0.025** | 0.450 |
| XTD-200 | ENG | 0.510 | 0.318 | 0.355 | 0.395 | **0.398** | 0.044 | 0.637 |
| XTD-200 | SEA | 0.017 | 0.013 | 0.012 | 0.014 | 0.011 | **0.037** | 0.381 |
| CVQA | EN | 0.395 | 0.354 | 0.353 | 0.350 | 0.338 | 0.271 | 0.538 |
| CVQA | LOCAL | 0.263 | 0.259 | 0.260 | 0.258 | 0.265 | 0.264 | 0.504 |
| **MEAN** | **ENG** | **0.448** | 0.308 | 0.335 | 0.367 | **0.371** | 0.079 | **0.624** |
| **MEAN** | **SEA** | 0.068 | 0.065 | 0.066 | 0.066 | 0.066 | **0.091** | **0.457** |

## Findings

1. **CC12M keeps English, transfers nothing to SEA — the mirror image of v1.** SEA sits at
   **0.066** across every epoch, statistically identical to the init's 0.068 (it does not move
   *at all*). Meanwhile English recovers from an early dip back to mean-ENG 0.371 / ImageNet
   0.400. Training-data language fully determines what KD transfers: no SEA data → no SEA gain.
2. **English dips then recovers, ending slightly below the init.** mean-ENG 0.448 (init) →
   0.308 (e8) → **0.371** (e32); ImageNet 0.426 → 0.296 → 0.400. Re-distilling a ViT-T from the
   B16 teacher on CC12M lands near — but not above — where the init already was. The ViT-T
   capacity caps English well below the B16 teacher (0.624).
3. **Non-Latin SEA is completely dead (0.001).** As with v1, the CLIP-BPE student cannot
   represent Thai/Myanmar script — but here even Latin SEA gets nothing, because there is no
   SEA supervision to begin with.
4. **CVQA ≈ random throughout** (LOCAL ~0.26, 4-way chance = 0.25); EN drifts down 0.354→0.338.

## Per-script / per-language breakdown (mean across Babel+XM3600+Flickr+XTD)

| group | e0 | cc e8 | cc e16 | cc e24 | cc e32 | v1 e32 | B16 teacher |
|---|---|---|---|---|---|---|---|
| **SEA-Latin** | 0.026 | 0.022 | 0.023 | 0.025 | 0.022 | **0.060** | 0.473 |
| **SEA-nonLatin** | 0.001 | 0.001 | 0.002 | 0.001 | 0.001 | **0.017** | 0.379 |
| en | 0.458 | 0.299 | 0.331 | 0.370 | 0.378 | 0.040 | 0.641 |
| id | 0.037 | 0.031 | 0.034 | 0.034 | 0.030 | 0.069 | 0.582 |
| ms | 0.031 | 0.025 | 0.027 | 0.027 | 0.026 | 0.067 | 0.528 |
| jv | 0.037 | 0.034 | 0.032 | 0.039 | 0.031 | 0.061 | 0.320 |
| vi | 0.008 | 0.006 | 0.006 | 0.009 | 0.009 | 0.056 | 0.540 |
| su | 0.038 | 0.032 | 0.031 | 0.030 | 0.028 | 0.052 | 0.258 |
| th | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.022 | 0.450 |
| my | 0.001 | 0.002 | 0.003 | 0.002 | 0.002 | 0.008 | 0.251 |

Every SEA language stays flat at its English-init floor; none of them move. The v1 SEA-blend
column (which *did* move them) is shown for contrast.

## Takeaway

The two runs bracket the trade-off cleanly, from the **same init/teacher/loss**:

- **CC12M (this run):** retains English (mean-ENG **0.371**), **zero** SEA transfer (SEA 0.066 = floor).
- **v1 SEA-blend:** forgets English (mean-ENG 0.079), gains modest SEA (SEA **0.091**).

So SEA capability comes only from SEA data in the KD set, and on this English init it costs
English. Neither run gets close to the B16 teacher (ENG 0.624 / SEA 0.457) — the ViT-T student
caps English, and CLIP-BPE structurally caps non-Latin SEA. The implied next step is the one
v1 already pointed to: **mix English (CC12M) + SEA data in a single KD run** to keep English
while acquiring SEA, ideally with a multilingual student tokenizer (XLM-V / SigLIP2) so non-Latin
scripts are representable at all.
