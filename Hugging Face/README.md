# Hugging Face — entrenamiento activo

Esta es la implementación activa del proyecto. Usa Transformers, PEFT/LoRA, `datasets` y PyTorch (MPS en Apple Silicon cuando está disponible) para entrenar y evaluar clasificadores de intención sobre Banking77.

## Preparar el entorno

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

Para las corridas de tres épocas se usan `qwen3_06b_banking77_epoch3.yaml` y `qwen35_08b_banking77_epoch3.yaml`.

## Evaluar y predecir

```bash
PYTHONPATH=src python -m finetuning.evaluate \
  --config configs/experiments/qwen35_08b_banking77.yaml \
  --adapter outputs/banking77/qwen35_08b/model

PYTHONPATH=src python -m finetuning.predict \
  --model outputs/banking77/qwen35_08b/model \
  --text "Why was my transfer declined?"
```

`configs/` contiene datasets y experimentos; `outputs/` contiene checkpoints, adapters, métricas y predicciones; `reports/` contiene tablas y matrices de confusión; `src/finetuning/` contiene el código ejecutable.
