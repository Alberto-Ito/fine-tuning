# Banking77 — two epochs of LoRA fine-tuning

- Base model: `Qwen/Qwen3-0.6B`
- Best checkpoint: `/Users/albertoito/Code/fireworks/fine-tuning/Qwen3-0.6B/outputs/banking77/checkpoints/two_epochs/checkpoint-2250`
- Completed global steps: 2,252
- Steps completed in the second-epoch run: 1,126
- Training examples per epoch: 9,002
- Mean training loss during the second epoch: 1.0275
- Training loss for the final logged block: 0.7469
- Validation loss: 0.5837
- Test loss: 0.5535
- Test accuracy: 0.8506
- Test macro-precision: 0.8559
- Test macro-recall: 0.8506
- Test macro-F1: 0.8509

## Comparison with one epoch

| Metric | One epoch | Two epochs | Change |
|---|---:|---:|---:|
| Validation loss | 0.7242 | 0.5837 | -0.1405 |
| Validation accuracy | 0.8132 | 0.8521 | +0.0390 |
| Validation macro-F1 | 0.8041 | 0.8394 | +0.0354 |
| Test loss | 0.7046 | 0.5535 | -0.1511 |
| Test accuracy | 0.8166 | 0.8506 | +0.0341 |
| Test macro-F1 | 0.8169 | 0.8509 | +0.0340 |

## Second-epoch validation progression

| Additional examples | Global step | Validation loss | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| ~1,000 | 1,250 | 0.7251 | 0.8112 | 0.7973 |
| ~2,000 | 1,375 | 0.7048 | 0.8092 | 0.7989 |
| ~3,000 | 1,500 | 0.6510 | 0.8412 | 0.8295 |
| ~4,000 | 1,625 | 0.6386 | 0.8332 | 0.8214 |
| ~5,000 | 1,750 | 0.6260 | 0.8322 | 0.8236 |
| ~6,000 | 1,875 | 0.5988 | 0.8362 | 0.8234 |
| ~7,000 | 2,000 | 0.5907 | 0.8442 | 0.8324 |
| ~8,000 | 2,125 | 0.5863 | 0.8482 | 0.8367 |
| ~9,000 | 2,250 | 0.5837 | 0.8521 | 0.8394 |

## Overfitting assessment

There is no evidence of overall overfitting after the second epoch. Training
loss decreased, validation and test loss both decreased, and accuracy and
macro-F1 improved on both held-out splits. Some intermediate checkpoints
showed short-lived metric regressions, but the final trend recovered and the
last scheduled checkpoint was the best one.

## Resource usage

- Total second-epoch loop time, including nine evaluations: 77.1 min
- Estimated optimization time: 55.9 min
- Peak allocated MPS memory: 1.124 GB
- Peak MPS driver memory: 2.937 GB
- Peak process RSS memory: 1.430 GB
- Checkpoint size: 289 MB
- Final adapter and tokenizer size: 20 MB

Per-class details are stored in `outputs/banking77/metrics/two_epochs/`.
Individual predictions are stored in `outputs/banking77/predictions/two_epochs/`.
The confusion matrix is stored in `reports/tables/banking77_two_epochs_confusion_matrix.csv`.
