# Fine-tuning local con Hugging Face, PyTorch y MPS

Esta carpeta reproduce la tarea SFT de `../mlx` sobre los mismos JSONL de
`../data`, sin modificarlos. Conserva batch 4, learning rate `1e-5`, longitud
máxima 512, pérdida sólo sobre la respuesta del asistente, LoRA rank 8 sobre
las últimas cuatro capas, evaluación cada 200 pasos y checkpoints cada 250.
El entrenamiento tiene un límite duro de 4.000 pasos.

El modelo MLX original (`mlx-community/llama2-13b-qnt4bit`) usa una
cuantización propia de MLX que PyTorch no puede cargar. El Llama 2 13B HF en
FP16 requiere aproximadamente 26 GB sólo para sus pesos, antes de activaciones
y optimizador, y no cabe en los 24 GB unificados de esta Mac. Por eso esta
reproducción usa `Qwen/Qwen2.5-0.5B-Instruct`, un modelo causal instructivo que
permite ejecutar realmente el mismo procedimiento con PyTorch/MPS. No es una
comparación directa de calidad entre modelos.

## Ejecución

```bash
cd "fine-tuning/mac-local/hugging face"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Piloto obligatorio
python train_hf.py --max-steps 30 --output-dir adapters-pilot

# Corrida completa, exactamente hasta 4.000 pasos
python train_hf.py --max-steps 4000 --output-dir adapters-4000

# Evaluación exacta de un checkpoint
python evaluate_sentiment.py --checkpoint adapters-4000/checkpoint-1000
```

Los resultados de la corrida ejecutada están en `result_0.4_epoch.md`. Incluyen
las pérdidas hasta 4000, precisión sobre 60 casos para los 16 checkpoints y la
distinción entre el mejor checkpoint por loss y por accuracy. Los detalles del
entorno están en `setup_hf.md`.
