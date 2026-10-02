# Banking77 Base-Model Evaluation

The dedicated virtual environment is `.venv/`. Each script automatically downloads its Hugging Face model with `from_pretrained`, loads `PolyAI/banking77`, and evaluates all 3,080 examples in the `test` split.

## Instalación

```bash
cd "fine-tuning/Base Model"
.venv/bin/python -m pip install -r requirements.txt
```

If network access is restricted, run this step on a machine with access to PyPI and Hugging Face. Models are stored in the Hugging Face cache; they do not need to be copied into these folders.

## Ejecución

```bash
.venv/bin/python Qwen3-0.6B/evaluate_base_model.py
.venv/bin/python Qwen3.5-0.8B/evaluate_base_model.py
```

Each run creates `results/metrics.json` and `results/predictions.jsonl` inside the model directory. Classification is zero-shot: the model receives the list of 77 labels and is instructed to return only the numeric ID. Token counts cover the prompt and generated tokens.
