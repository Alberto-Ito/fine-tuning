# Evaluación de modelos base en Banking77

El entorno virtual dedicado está en `.venv/`. Cada script descarga automáticamente su modelo desde Hugging Face mediante `from_pretrained`, descarga/carga `PolyAI/banking77` y evalúa los 3.080 ejemplos del split `test`.

## Instalación

```bash
cd "fine-tuning/Base Model"
.venv/bin/python -m pip install -r requirements.txt
```

Si la red está restringida, ejecutar este paso en una máquina con acceso a PyPI/Hugging Face. Los modelos quedarán en la caché de Hugging Face; no hace falta copiarlos manualmente a las carpetas.

## Ejecución

```bash
.venv/bin/python Qwen3-0.6B/evaluate_base_model.py
.venv/bin/python Qwen3.5-0.8B/evaluate_base_model.py
```

Cada ejecución genera `results/metrics.json` y `results/predictions.jsonl` dentro de la carpeta del modelo. La clasificación es zero-shot: se presenta al modelo la lista de 77 etiquetas y se le pide devolver únicamente el ID numérico. Los conteos de tokens se miden sobre el prompt y los tokens generados.
