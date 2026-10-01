# Banking77 pilot: Qwen3.5-4B v1

This pilot targets the Hugging Face model `Qwen/Qwen3.5-4B`, which corresponds
to the Microsoft Foundry catalog model `qwen-qwen3.5-4b`, version `1`.

## Current deployment constraint

As of 2026-10-01, Microsoft documents **Qwen 3.5 4B as an exception to LoRA
adapter deployment** on Fireworks in Foundry. The catalog entry exists at
version 1 and custom full-weight import is supported, but uploading a LoRA
adapter for this base model is not currently supported.

Official references:

- [Import custom models into Microsoft Foundry with Fireworks](https://learn.microsoft.com/en-us/azure/foundry/how-to/fireworks/import-custom-models)
- [Qwen3.5-4B catalog entry](https://ai.azure.com/catalog/models/qwen-qwen3.5-4b)
- [Qwen/Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)

The local pilot is still useful for checking memory use, convergence, and
Banking77 classification quality while support is evaluated.

## Run the 100-example local pilot

From the repository's `fine-tuning` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

finetune-train \
  --config configs/experiments/qwen35_4b_v1_banking77_pilot100.yaml
```

The configuration samples at most 100 examples independently from the train,
validation, and test splits, trains for one epoch, and stores all new artifacts
under `outputs/pilots/banking77_100/qwen35_4b_v1/` and
`reports/pilots/banking77_100/qwen35_4b_v1/`.

The conservative per-device batch size of 1 with four-step gradient
accumulation keeps the effective training batch size aligned with the existing
0.8B pilot while reducing peak device memory. Adjust `dtype` to `float32` only
if the selected device cannot execute `bfloat16`; this substantially increases
memory use.

## Important artifact distinction

The maintained repository pipeline uses
`AutoModelForSequenceClassification` and creates a PEFT adapter with
`task_type: SEQ_CLS` plus a saved classification head. Foundry's Qwen catalog
endpoint is a generative vLLM endpoint. Therefore this pilot's adapter should
not be treated as a deployable chat-completions adapter, independently of the
current Qwen3.5-4B LoRA restriction.

Do not upload the generated adapter as a Foundry LoRA asset. Before a Foundry
deployment, choose one of these paths:

1. Wait until Microsoft lists Qwen3.5-4B as LoRA-compatible, then build a
   causal-language-model SFT adapter whose output is one of the 77 canonical
   intent names.
2. Use a smaller base model that Foundry currently lists as LoRA-compatible.
3. Build and validate a merged full-weight causal model, then import it using
   Foundry's `Full weight model` workflow. This is a different training and
   serving contract from the sequence-classification pilot in this repository.

When LoRA support becomes available, the upload directory must contain at
least `adapter_config.json` and `adapter_model.safetensors` (or
`adapter_model.bin`), and the base model selected in the portal must exactly
match the adapter's base model.
