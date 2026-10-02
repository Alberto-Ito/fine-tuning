# Local Fine-Tuning with Hugging Face, PyTorch, and MPS

This folder reproduces the SFT task from `../mlx` using the same JSONL files in
`../data`, without modifying them. It uses batch size 4, learning rate `1e-5`,
maximum length 512, loss only on the assistant response, LoRA rank 8 on the
last four layers, evaluation every 200 steps, and checkpoints every 250 steps.
Training is capped at 4,000 steps.

The original MLX model (`mlx-community/llama2-13b-qnt4bit`) uses MLX-specific
quantization that PyTorch cannot load. The 13B Llama 2 HF model in FP16 needs
approximately 26 GB just for weights, before activations and the optimizer, and
does not fit in this Mac's 24 GB unified memory. This reproduction therefore
uses `Qwen/Qwen2.5-0.5B-Instruct`, a causal instruct model that can run the same
procedure with PyTorch/MPS. It is not a direct quality comparison between the
architectures.

## Ejecución

```bash
cd "fine-tuning/Legacy Mac Experiments/hugging face"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Piloto obligatorio
python train_hf.py --max-steps 30 --output-dir adapters-pilot

# Corrida completa, exactamente hasta 4.000 pasos
python train_hf.py --max-steps 4000 --output-dir adapters-4000

# Evaluación exacta de un checkpoint
python evaluate_sentiment.py --checkpoint adapters-4000/checkpoint-1000
```

Results from the completed run are in `result_0.4_epoch.md`. They include loss
through step 4000, accuracy on 60 cases for all 16 checkpoints, and the
distinction between the best checkpoint by loss and by accuracy. Environment
details are in
entorno están en `setup_hf.md`.
