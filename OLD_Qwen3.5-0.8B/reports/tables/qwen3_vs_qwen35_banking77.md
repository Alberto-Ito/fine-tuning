# Banking77 — Qwen3-0.6B vs Qwen3.5-0.8B

Both models were fine-tuned for two epochs with LoRA on the same Banking77
train/validation/test split (seed 42, effective training batch size 8,
maximum sequence length 128).

## Quality

| Metric | Qwen3-0.6B | Qwen3.5-0.8B | Qwen3.5 change |
|---|---:|---:|---:|
| Test loss | 0.5535 | **0.4934** | **-0.0601 (-10.86%)** |
| Test accuracy | 85.06% | **86.88%** | **+1.82 pp** |
| Test macro-precision | 85.59% | **87.40%** | **+1.81 pp** |
| Test macro-recall | 85.06% | **86.88%** | **+1.82 pp** |
| Test macro-F1 | 85.09% | **86.89%** | **+1.80 pp** |

## Operational benchmark

The two adapters were loaded in separate clean processes and measured with the
same protocol: Apple MPS, bfloat16, test split, batch size 8, 3 warm-up batches,
382 measured batches, and 3,056 measured requests. Tokens/request differs by
0.01 because each model uses its own tokenizer.

| Metric | Qwen3-0.6B | Qwen3.5-0.8B | Qwen3.5 change |
|---|---:|---:|---:|
| Input tokens/request, mean | 12.61 | 12.60 | -0.01 |
| Requests/s | **6.11** | 3.04 | **-50.31%** |
| Input tokens/s | **77.02** | 38.24 | **-50.35%** |
| Batch latency p50 | **1,309.80 ms** | 2,633.93 ms | **+101.09%** |
| Batch latency p95 | **1,327.00 ms** | 2,682.31 ms | **+102.13%** |
| Peak process RSS | **0.607 GB** | 0.722 GB | **+18.95%** |
| Peak MPS allocated | **1.117 GB** | 1.592 GB | **+42.52%** |
| Peak MPS driver | **1.380 GB** | 2.284 GB | **+65.51%** |
| Model + adapter load time | 2.99 s | **2.06 s** | **-31.09%** |

Qwen3.5-0.8B therefore trades operational efficiency for quality: it adds
1.82 percentage points of test accuracy and 1.80 points of macro-F1, while the
standalone MPS benchmark processes about half as many requests per second and
uses 42.5% more allocated MPS memory.

The Qwen3.5 hybrid DeltaNet layers used the correct PyTorch reference kernels
because optimized `causal_conv1d` and `flash-linear-attention` kernels were not
available for this MPS environment. Consequently, these throughput results
describe this Mac setup and should not be generalized to optimized CUDA
serving stacks.

## Recommendation

- Prefer **Qwen3.5-0.8B** when the ~1.8-point quality gain matters more than
  latency and throughput.
- Prefer **Qwen3-0.6B** for local Mac serving when throughput, latency, and
  memory efficiency are the priority.
