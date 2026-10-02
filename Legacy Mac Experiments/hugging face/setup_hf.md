# Setup de Hugging Face + PyTorch/MPS

Entorno creado con Python 3.9.6 en `.venv`. Instalación ejecutada:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers datasets peft accelerate sentencepiece safetensors
```

Dependencias directas instaladas:

| Paquete | Versión | Uso |
|---|---:|---|
| `torch` | 2.8.0 | Entrenamiento y backend MPS |
| `transformers` | 4.57.6 | Modelo, tokenizer y Trainer |
| `datasets` | 4.5.0 | Utilidades de datos HF |
| `peft` | 0.17.1 | Adaptadores LoRA |
| `accelerate` | 1.10.1 | Selección e integración del dispositivo MPS |
| `sentencepiece` | 0.2.2 | Tokenizers compatibles |
| `safetensors` | 0.7.0 | Lectura y escritura segura de pesos |

`pip` resolvió además las dependencias transitivas registradas en `requirements-lock.txt`.
El modelo base se descarga de Hugging Face la primera vez que se ejecuta el entrenamiento.

Comprobación de MPS:

```bash
python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
```
