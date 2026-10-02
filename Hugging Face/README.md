# Hugging Face — Active Training Pipeline

This is the active project implementation. It uses Transformers, PEFT/LoRA, `datasets`, and PyTorch (MPS on Apple Silicon when available) to train and evaluate Banking77 intent classifiers.

## Environment setup

```bash
cd "fine-tuning/Hugging Face"
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Entrenar

```bash
PYTHONPATH=src python -m finetuning.train \
  --config configs/experiments/qwen3_06b_banking77.yaml

PYTHONPATH=src python -m finetuning.train \
  --config configs/experiments/qwen35_08b_banking77.yaml
```

Use `qwen3_06b_banking77_epoch3.yaml` and `qwen35_08b_banking77_epoch3.yaml` for three-epoch runs.

## Evaluar y predecir

```bash
PYTHONPATH=src python -m finetuning.evaluate \
  --config configs/experiments/qwen35_08b_banking77.yaml \
  --adapter outputs/banking77/qwen35_08b/model

PYTHONPATH=src python -m finetuning.predict \
  --model outputs/banking77/qwen35_08b/model \
  --text "Why was my transfer declined?"
```

`configs/` contains dataset and experiment definitions; `outputs/` contains checkpoints, adapters, metrics, and predictions; `reports/` contains tables and confusion matrices; `src/finetuning/` contains the executable package.
