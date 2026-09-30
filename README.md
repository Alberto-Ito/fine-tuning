# Configurable fine-tuning

The canonical implementation now lives under `src/finetuning/`. Model and
dataset choices are configuration-driven, so the same `train`, `evaluate`, and
`predict` entry points work for Qwen3, Qwen3.5, and future datasets.

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

finetune-train --config configs/experiments/qwen3_06b_banking77.yaml
finetune-train --config configs/experiments/qwen35_08b_banking77.yaml
```

The same commands are available as modules:

```bash
PYTHONPATH=src python -m finetuning.train --config configs/experiments/qwen35_08b_banking77.yaml
PYTHONPATH=src python -m finetuning.evaluate --config configs/experiments/qwen35_08b_banking77.yaml --adapter outputs/banking77/qwen35_08b/model
PYTHONPATH=src python -m finetuning.predict --model outputs/banking77/qwen35_08b/model --text "Why was my transfer declined?"
```

The previous per-model directories remain available as historical/reference
experiments. New work should use the shared `src/finetuning` package.
