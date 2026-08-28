| Model | Image params | Text params | Image GFLOPs | Text GFLOPs | Total GFLOPs | Image ms (bs=1) | Text ms (bs=1) | img/s (bs=1) | txt/s (bs=1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyCLIP | 8.28M | 15.17M | 3.18 | 0.36 | 3.54 | 12.52 | 2.40 | 81 | 414 |
| MobileCLIP2 | 11.41M | 63.43M | 4.77 | 3.88 | 8.64 | 41.57 | 20.03 | 24 | 50 |
| SEA-CLIP-Tiny | 5.62M | 40.49M | 2.15 | 3.27 | 5.42 | 10.40 | 13.73 | 96 | 72 |
| MetaCLIP-2 B/16 (teacher) | 86.19M | 499.77M | 33.70 | 5.81 | 39.51 | 70.24 | 20.25 | 14 | 49 |

- GFLOPs per sample, torch.utils.flop_counter — true FLOPs (1 MAC = 2 FLOPs).
- Each model at its own native input resolution and context length — TinyCLIP [3, 224, 224] / 77 tokens; MobileCLIP2 [3, 256, 256] / 77 tokens; SEA-CLIP-Tiny [3, 224, 224] / 77 tokens; MetaCLIP-2 B/16 (teacher) [3, 224, 224] / 77 tokens.
- Parameter counts include each tower's final embedding projection head (0.00M, 0.10M, 0.13M, 0.39M image / 0.13M, 0.20M, 0.26M text).
- Latency = median of 30 runs after 5 warmup passes, fp32, on cpu.
- Throughput = batch size / mean batch time at bs=1, same runs and precision.
