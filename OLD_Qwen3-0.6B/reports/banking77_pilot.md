# Pilot results: Banking77 + Qwen3-0.6B + LoRA

## Configuration

- Backend: Hugging Face Transformers + PEFT + PyTorch MPS
- Model: `Qwen/Qwen3-0.6B`
- Method: LoRA for sequence classification
- Precision: `bfloat16`
- Steps: 50
- Per-device batch size: 4
- Gradient accumulation: 2
- Effective batch size: 8
- Maximum sequence length: 128
- Classes: 77
- Trainable parameters: 1,225,728 (0.2052%)

## Observed resource usage

- Peak allocated MPS memory: 1.237 GB
- Peak MPS driver memory: 2.929 GB
- Peak process RSS memory: 0.747 GB
- Estimated optimization time per step: 3.58 seconds
- Full validation time: 113–169 seconds for 1,001 examples
- Cached base-model size: 1.4 GB
- Combined size of both pilot checkpoints: 58 MB
- Final adapter and tokenizer size: 20 MB

## Step 50 metrics

- Mean training loss: 10.9809
- Validation loss: 5.3126
- Accuracy: 0.0120
- Macro-F1: 0.0046

The pilot processed approximately 400 examples, or 4.44% of one epoch. Its
accuracy remains near random chance for 77 classes and should not be treated as
the final model performance.

## Conclusion

The Mac has sufficient memory for this LoRA training run. The first `float16`
attempt produced `NaN` gradients; switching to `bfloat16` resolved the issue
and completed every step without numerical instability.

One epoch over the 9,002 training examples requires approximately 1,126 steps
with an effective batch size of 8. At the observed rate, optimization alone
would take about 67 minutes. Including evaluation and possible thermal
throttling on the fanless MacBook Air, a realistic estimate is 75–100 minutes.
