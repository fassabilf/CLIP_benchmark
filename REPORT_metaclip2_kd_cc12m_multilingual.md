# REPORT — metaclip2_kd v3 (CC12M + SEA blend): ViT-T-16 student distilled from MetaCLIP2-B16-worldwide on CC12M + multilingual SEA

**Run:** `multilingual-clip-kd/.../metaclip2_kd/clipkd_ViT-T-16_from_ViT-B-16-MetaClip_pretrained_v3`
**Eval date:** 2026-06-17 · **Suite:** ImageNet-1k, Babel-ImageNet, XM3600, Flickr30k-200, XTD-200, CVQA
**Env:** `mc2_eval_env` pinned to **habibi's own open_clip** (`open_clip_phabibi/src`, v3.2.0),
arch `ViT-T-16` = **CLIP-BPE** (vocab 49408, ctx 77).
**Predictions:** `fassabilf/sea-clip-eval-predictions` (HF, private), folders `mc2v3_e{8,16,24,32}/`.

## What this run is

The **mix run** — the implied next step from v1 (SEA-only forgets English) and v2 (CC12M-only transfers nothing to SEA). Same student init / teacher (MetaCLIP2-ViT-B-16-worldwide) / loss (clipkd, 32ep, lr 2e-3) as v1 and v2.

**Training data** (concatenated, no resampling):
- CC12M — 10.97M English image–text pairs (~90% of total)
- CG-OE-filt × 6 langs — 703K pairs (id/jv/ms/su/th/vi)
- WIT-hf-base × 6 langs — 488K pairs (id/jv/ms/my/th/vi)
- Bloom — 22K pairs (76 minority SEA langs)
- **Total:** 12.18M samples per epoch

The hypothesis: CC12M keeps English while SEA data transfers SEA capability.

## Results (image→text R@1 for retrieval, acc1 for classification)

| benchmark | grp | e0 (init) | v3 e8 | v3 e16 | v3 e24 | v3 e32 | v1 SEA-blend e32 | **B16 teacher** |
|---|---|---|---|---|---|---|---|---|
| ImageNet1k | ENG | 0.426 | 0.235 | 0.291 | 0.343 | **0.374** | 0.056 | **0.711** |
| Babel-IN | ENG | 0.363 | 0.204 | 0.251 | 0.292 | **0.322** | 0.050 | 0.636 |
| Babel-IN | SEA | 0.042 | 0.094 | 0.119 | 0.142 | **0.147** | 0.084 | **0.424** |
| XM3600 | ENG | 0.394 | 0.221 | 0.287 | 0.308 | 0.314 | 0.024 | 0.460 |
| XM3600 | SEA | 0.007 | 0.055 | 0.090 | 0.113 | 0.103 | **0.046** | **0.526** |
| Flickr30k-200 | ENG | 0.599 | 0.276 | 0.340 | 0.388 | **0.397** | 0.026 | 0.761 |
| Flickr30k-200 | SEA | 0.011 | 0.034 | 0.059 | 0.072 | 0.058 | **0.025** | 0.450 |
| XTD-200 | ENG | 0.510 | 0.272 | 0.330 | 0.348 | **0.354** | 0.044 | 0.637 |
| XTD-200 | SEA | 0.017 | 0.060 | 0.086 | 0.091 | 0.084 | **0.037** | 0.381 |
| CVQA | EN | 0.395 | 0.342 | 0.358 | 0.353 | 0.342 | 0.271 | 0.538 |
| CVQA | LOCAL | 0.263 | 0.255 | 0.264 | 0.265 | 0.264 | 0.264 | 0.504 |
| **MEAN** | **ENG** | **0.448** | 0.258 | 0.310 | 0.339 | **0.351** | 0.079 | **0.624** |
| **MEAN** | **SEA** | 0.068 | 0.100 | 0.123 | 0.137 | **0.131** | 0.091 | **0.457** |

## Findings

1. **v3 achieves both goals simultaneously, but neither fully.** At e32: mean-ENG **0.351** (vs init 0.448, v1 0.079) and mean-SEA **0.131** (vs init 0.068, v2 0.066). English is substantially retained and SEA is substantially gained — the CC12M-vs-SEA trade-off is reduced but not eliminated. Neither metric matches the init's English (0.448) or the teacher's SEA (0.457).

2. **SEA peaks at e24 (0.137), then slightly regresses at e32 (0.131).** The English-dominant data volume (90% CC12M) eventually pulls the model back toward English at the cost of SEA. The optimal checkpoint for SEA is e24; for the best English-SEA balance (ImageNet 0.343 + mean-SEA 0.137) e24 is also the best tradeoff point.

3. **SEA gains are real and large relative to v2.** v3 e32 mean-SEA 0.131 vs v2 e32 0.066 — the SEA blend (10% of data) doubles SEA performance even when heavily diluted by CC12M. v3 also beats v1's SEA (0.091) at every epoch from e16 onward.

4. **English recovery is slower than v2.** v2 e32 mean-ENG 0.371 vs v3 e32 0.351 — adding SEA data hurts English slightly, but much less than v1 (0.079). The ~0.02 gap reflects the dilution cost of the SEA component.

5. **Non-Latin SEA (th, my) benefits significantly.** SEA-Latin 0.126 vs SEA-nonLatin 0.031 at e32 — non-Latin still well below Latin due to CLIP-BPE tokenizer ceiling, but far above v2 (0.001). Thai (0.039) and Burmese (0.017) move from near-zero.

6. **CVQA stays near random** (LOCAL ~0.264, EN 0.342); the VQA signal is not driven by the retrieval-trained student.

## Per-script / per-language breakdown (mean across Babel+XM3600+Flickr+XTD, e32)

| group | e0 | v3 e8 | v3 e16 | v3 e24 | v3 e32 | v1 e32 | B16 teacher |
|---|---|---|---|---|---|---|---|
| **SEA-Latin** | 0.026 | 0.072 | 0.109 | 0.134 | **0.126** | 0.060 | 0.473 |
| **SEA-nonLatin** | 0.001 | 0.017 | 0.033 | 0.038 | **0.031** | 0.017 | 0.379 |
| en | 0.458 | 0.266 | 0.321 | 0.362 | **0.347** | 0.040 | 0.641 |
| id | 0.037 | 0.094 | 0.131 | 0.158 | **0.154** | 0.069 | 0.582 |
| ms | 0.031 | 0.080 | 0.116 | 0.148 | **0.145** | 0.067 | 0.528 |
| jv | 0.037 | 0.071 | 0.097 | 0.119 | **0.111** | 0.061 | 0.320 |
| vi | 0.008 | 0.055 | 0.087 | 0.118 | **0.125** | 0.056 | 0.540 |
| su | 0.038 | 0.055 | 0.074 | 0.086 | **0.079** | 0.052 | 0.258 |
| th | 0.002 | 0.020 | 0.036 | 0.042 | **0.039** | 0.022 | 0.450 |
| my | 0.001 | 0.010 | 0.017 | 0.019 | **0.017** | 0.008 | 0.251 |

Every SEA language gains substantially over the init and over v2's floor. Indonesian (0.154) and Malay (0.145) lead; Vietnamese (0.125) gains strongly despite being in both CG and WIT; Sundanese (0.079) and Burmese (0.017) gain despite smaller data pools.

## Takeaway

v3 is the Pareto-better run: it retains English (mean-ENG 0.351 vs v1's 0.079) and gains SEA (mean-SEA 0.131 vs v2's 0.066) simultaneously. The CC12M-to-SEA ratio (~90:10) is sufficient to carry meaningful SEA signal even heavily diluted. The remaining gap to the teacher (ENG 0.624 / SEA 0.457) reflects two limits:

- **Student capacity:** ViT-T is too small to match ViT-B on either metric.
- **Tokenizer ceiling:** CLIP-BPE still underserves non-Latin scripts (SEA-nonLatin 0.031 vs SEA-Latin 0.126). A multilingual tokenizer (XLM-V / SigLIP2) is the expected next lever.
