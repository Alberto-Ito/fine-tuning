# Banking77 — one epoch of LoRA fine-tuning

- Base model: `Qwen/Qwen3-0.6B`
- Best checkpoint: `/Users/albertoito/Code/fireworks/fine-tuning/Qwen3-0.6B/outputs/banking77/checkpoints/one_epoch/checkpoint-1125`
- Completed steps: 1,126
- Training examples: 9,002
- Mean training loss: 3.5744
- Training loss for the final logged block: 1.3081
- Validation loss: 0.7242
- Test loss: 0.7046
- Test accuracy: 0.8166
- Test macro-precision: 0.8250
- Test macro-recall: 0.8166
- Test macro-F1: 0.8169

## Validation progression

| Examples seen | Step | Validation loss | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| 1.000 | 125 | 4.3701 | 0.0280 | 0.0065 |
| 2.000 | 250 | 2.7127 | 0.3357 | 0.2691 |
| 3.000 | 375 | 1.4591 | 0.6074 | 0.5816 |
| 4.000 | 500 | 1.0952 | 0.7023 | 0.6822 |
| 5.000 | 625 | 0.9348 | 0.7572 | 0.7453 |
| 6.000 | 750 | 0.8259 | 0.7812 | 0.7680 |
| 7.000 | 875 | 0.7563 | 0.7992 | 0.7892 |
| 8.000 | 1.000 | 0.7274 | 0.8062 | 0.7959 |
| 9.000 | 1.125 | 0.7242 | 0.8132 | 0.8041 |

Hugging Face also saved `checkpoint-1126` after the final minibatch of the
epoch. The best validation result came from `checkpoint-1125`.

## Resource usage

- Total training-loop time, including nine evaluations: 65.9 min
- Estimated optimization time: 47.9 min
- Peak allocated MPS memory: 1.237 GB
- Peak MPS driver memory: 3.039 GB
- Peak process RSS memory: 1.442 GB
- Checkpoint size: 289 MB
- Final adapter and tokenizer size: 20 MB

## Classes with the lowest test F1

| Class | F1 | Support |
|---|---:|---:|
| `topping_up_by_card` | 0.5405 | 40 |
| `cash_withdrawal_not_recognised` | 0.6098 | 40 |
| `pending_transfer` | 0.6301 | 40 |
| `balance_not_updated_after_bank_transfer` | 0.6329 | 40 |
| `beneficiary_not_allowed` | 0.6486 | 40 |

Per-class details are stored in `outputs/banking77/metrics/one_epoch/`.
Individual predictions are stored in `outputs/banking77/predictions/one_epoch/`.
The confusion matrix is stored in `reports/tables/banking77_confusion_matrix.csv`.
