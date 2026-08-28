| Model | Image params | Text params | Image GFLOPs | Text GFLOPs | Total GFLOPs | Image ms (bs=1) | Text ms (bs=1) | img/s (bs=64) | txt/s (bs=64) | Peak mem MB (bs=64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyCLIP | 8.28M | 15.17M | 3.57 | 0.38 | 3.96 | 5.63 | 2.48 | 11046 | 28628 | 248 |
| MobileCLIP2 | 11.41M | 63.43M | 4.79 | 3.88 | 8.66 | 14.31 | 5.61 | 1828 | 10775 | 858 |
| SEA-CLIP-Tiny | 5.62M | 40.49M | 2.51 | 3.38 | 5.89 | 5.48 | 5.60 | 10836 | 10422 | 322 |
| MetaCLIP-2 B/16 (teacher) | 86.19M | 499.77M | 35.13 | 5.96 | 41.09 | 6.17 | 6.65 | 2998 | 8811 | 2758 |

- GFLOPs per sample, torch.utils.flop_counter — true FLOPs (1 MAC = 2 FLOPs).
- Each model at its own native input resolution and context length — TinyCLIP [3, 224, 224] / 77 tokens; MobileCLIP2 [3, 256, 256] / 77 tokens; SEA-CLIP-Tiny [3, 224, 224] / 77 tokens; MetaCLIP-2 B/16 (teacher) [3, 224, 224] / 77 tokens.
- Parameter counts include each tower's final embedding projection head (0.00M, 0.10M, 0.13M, 0.39M image / 0.13M, 0.20M, 0.26M text).
- Latency = median of 100 runs after 20 warmup passes, amp, on NVIDIA A100-SXM4-40GB.
- Throughput = batch size / mean batch time at bs=64, same runs and precision.
- Peak memory = torch.cuda.max_memory_allocated over the measured bs=64 passes (weights resident + activations of the larger tower), after warmup.
