# Banking77 — Qwen3.5-0.8B, two epochs of LoRA fine-tuning

- Base model: `Qwen/Qwen3.5-0.8B`
- Best checkpoint: `/Users/albertoito/Code/fireworks/fine-tuning/Qwen3.5-0.8B/outputs/banking77/checkpoints/two_epochs/checkpoint-2125`
- Completed steps: 2252
- Test loss: 0.4934
- Test accuracy: 0.8688
- Test macro-F1: 0.8689

## Operational metrics

- Input tokens/request (mean): 12.60
- Requests/s: 4.41
- Input tokens/s: 55.53
- Batch latency p50/p95: 1815.74 / 1823.50 ms
- Model load time: 1.36 s
- Peak process RSS: 1.797 GB
- Peak MPS allocated/driver: 1.594 / 5.100 GB

Full metrics: `outputs/banking77/metrics/two_epochs/run_summary.json`.
