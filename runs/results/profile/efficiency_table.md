| Model | Image params | Text params | Image GFLOPs | Text GFLOPs | Total GFLOPs | Image ms (bs=1) | Text ms (bs=1) | img/s (bs=1024) | txt/s (bs=1024) | Peak mem MB (bs=1024) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyCLIP | 8.15M | 2.39M | 3.57 | 0.38 | 3.96 | 5.31 | 2.34 | 14465 | 112685 | 2278 |
| MobileCLIP2 | 11.41M | 37.87M | 4.79 | 3.88 | 8.66 | 15.81 | 5.73 | 1929 | 14371 | 9258 |
| SEA-CLIP-Tiny | 5.52M | 21.32M | 2.51 | 3.38 | 5.89 | 5.97 | 6.27 | 13199 | 18551 | 1739 |
| MetaCLIP-2 B/16 (teacher) | 85.80M | 37.87M | 35.13 | 5.96 | 41.09 | 6.07 | 6.96 | 3426 | 15122 | 7745 |

- GFLOPs per sample, torch.utils.flop_counter — true FLOPs (1 MAC = 2 FLOPs).
- Each model at its own native input resolution and context length — TinyCLIP [3, 224, 224] / 77 tokens; MobileCLIP2 [3, 256, 256] / 77 tokens; SEA-CLIP-Tiny [3, 224, 224] / 77 tokens; MetaCLIP-2 B/16 (teacher) [3, 224, 224] / 77 tokens.
- Parameter counts exclude each tower's final embedding projection head (0.00M, 0.10M, 0.13M, 0.39M image / 0.13M, 0.20M, 0.26M text).
- Text counts exclude the token-embedding look-up table (TinyCLIP 12.65M, MobileCLIP2 25.30M, SEA-CLIP-Tiny 18.97M, MetaCLIP-2 B/16 (teacher) 461.63M). TinyCLIP's published counts exclude it.
- Latency = median of 100 runs after 20 warmup passes, amp, on NVIDIA A100-SXM4-40GB.
- Throughput = batch size / mean batch time at bs=1024, same runs and precision.
- Peak memory = torch.cuda.max_memory_allocated over the measured bs=1024 passes (weights resident + activations of the larger tower), after warmup.
