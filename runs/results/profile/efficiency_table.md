| Model | Image params | Text params | Image GFLOPs | Text GFLOPs | Total GFLOPs | Image ms (bs=1) | Text ms (bs=1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyCLIP | 8.15M | 15.04M | 3.57 | 0.38 | 3.96 | 4.59 | 1.93 |

- GFLOPs per sample, torch.utils.flop_counter — true FLOPs (1 MAC = 2 FLOPs).
- Input [3, 224, 224] image / 77-token text (each model at its own native resolution and context length).
- Parameter counts exclude each tower's final embedding projection head (0.13M image / 0.13M text).
- Latency = median of 50 runs after 10 warmup passes, amp, on NVIDIA GeForce RTX 3050 Laptop GPU.
