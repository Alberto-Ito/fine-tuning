# Qwen3-0.6B fine-tuning

This project contains a reproducible pipeline for fine-tuning and evaluating
`Qwen/Qwen3-0.6B` on two independent intent-classification tasks:

- `PolyAI/banking77`
- `clinc/clinc_oos`

## Structure

```text
Qwen3-0.6B/
├── configs/
│   ├── datasets/       # Dataset-specific settings
│   └── experiments/    # Hyperparameters for each run
├── data/
│   ├── raw/            # Original, immutable downloads
│   └── processed/      # Transformed data ready for training
├── notebooks/          # Interactive exploration and analysis
├── outputs/
│   ├── banking77/      # Checkpoints, logs, metrics, and predictions
│   └── clinc_oos/      # for the corresponding experiment
├── reports/
│   ├── figures/        # Comparison charts
│   └── tables/         # Result tables
├── scripts/            # Pipeline entry points
├── src/
│   └── qwen3_finetuning/
│       ├── data/       # Downloading, validation, and preprocessing
│       ├── training/   # Model, tokenization, and training
│       ├── evaluation/ # Metrics, inference, and comparisons
│       └── utils/      # Shared utilities
└── tests/              # Unit and integration tests
```

## Intended workflow

1. Download each dataset into `data/raw/<dataset>/`.
2. Normalize it and save it to `data/processed/<dataset>/`.
3. Define settings in `configs/datasets/` and `configs/experiments/`.
4. Train and save each run under `outputs/<dataset>/`.
5. Evaluate the model and consolidate comparisons in `reports/`.

Datasets, checkpoints, and other large artifacts remain local and are not
versioned. The `.gitkeep` files preserve the empty directory structure only.

## Introductory notebook

`notebooks/01_exploracion_dataset.ipynb` explains how to load a dataset from
Hugging Face, inspect its splits and columns, review label distributions, read
examples, and run basic quality checks. It uses Banking77 by default and can
switch to CLINC OOS by changing one variable.

## Banking77 Hugging Face pilot

The first experiment uses `transformers`, `datasets`, `Trainer`, PyTorch MPS,
and PEFT LoRA. Its configuration is stored in
`configs/experiments/banking77_pilot.yaml` and can be run with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_banking77.py
```

The pilot performs 50 optimization steps. It stores checkpoints, the final
adapter, logs, and metrics under `outputs/banking77/`.
