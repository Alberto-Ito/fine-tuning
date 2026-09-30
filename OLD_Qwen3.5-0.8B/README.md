# Qwen3.5-0.8B fine-tuning

Reproducible two-epoch LoRA fine-tuning of `Qwen/Qwen3.5-0.8B` on Banking77,
adapted from the adjacent `Qwen3-0.6B` experiment.

The run records classification quality plus operational metrics from batched
test inference: input tokens/request, requests/s, input tokens/s, batch latency
p50/p95, model-load time, and peak process/MPS memory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_banking77.py --config configs/experiments/banking77_2epochs.yaml
```

Artifacts are written below `outputs/banking77/`; the human-readable result is
written to `reports/tables/banking77_two_epochs.md`.
