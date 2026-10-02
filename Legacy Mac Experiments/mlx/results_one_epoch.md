# Fine-tuning results through one epoch

## Configuration

- Model: `mlx-community/llama2-13b-qnt4bit`
- Training data: 10,000 examples
- Method: QLoRA, 4 trainable layers, `mask-prompt`, gradient checkpointing
- Learning rate: `1e-5`
- Batch size: 4
- Maximum sequence length: 512 tokens
- One epoch: 2,500 iterations × 4 examples = 10,000 examples
- Peak memory usage: 8.717 GB
- Evaluation set: 60 held-out cases; 40 negative, 10 positive, and 10 neutral

## Loss progression

| Iteration | Epoch fraction | Train loss | Validation loss |
|---:|---:|---:|---:|
| 1 | 0.00 | — | 6.123 |
| 200 | 0.08 | 2.333 | 2.312 |
| 400 | 0.16 | 0.548 | 0.492 |
| 600 | 0.24 | 0.464 | 0.459 |
| 800 | 0.32 | 0.459 | 0.454 |
| 1,000 | 0.40 | 0.466 | 0.452 |
| 1,200 | 0.48 | 0.450 | 0.448 |
| 1,400 | 0.56 | 0.445 | 0.444 |
| 1,600 | 0.64 | 0.446 | 0.442 |
| 1,800 | 0.72 | 0.443 | 0.439 |
| 2,000 | 0.80 | 0.446 | 0.437 |
| 2,200 | 0.88 | 0.439 | 0.436 |
| 2,400 | 0.96 | 0.432 | 0.431 |
| 2,500 | 1.00 | 0.430 | 0.429 |

Validation loss continued to decrease throughout the epoch and did not diverge
from training loss. However, nearly all of the reduction occurred before 0.25
epoch.

## Exact accuracy by checkpoint

| Checkpoint | Examples seen | Epoch fraction | Accuracy | Valid JSON |
|---:|---:|---:|---:|---:|
| 250 | 1,000 | 0.10 | 49/60 = 81.7% | 60/60 |
| 500 | 2,000 | 0.20 | 43/60 = 71.7% | 60/60 |
| **1,000** | **4,000** | **0.40** | **57/60 = 95.0%** | **60/60** |
| 1,750 | 7,000 | 0.70 | 48/60 = 80.0% | 60/60 |
| 2,500 | 10,000 | 1.00 | 50/60 = 83.3% | 60/60 |

The best observed result is `0001000_adapters.safetensors`. Its three errors
were one negative classified as positive, one positive classified as negative,
and one positive classified as neutral. This result should be considered
preliminary because the 60-case evaluation set is small and imbalanced.

## Conclusions

1. Lower loss did not guarantee higher classification accuracy. The final
   checkpoint had lower loss but worse accuracy than the 0.4-epoch checkpoint.
2. Training through a complete epoch did not produce the best model for this
   evaluation set.
3. Every checkpoint generated valid JSON for all 60 cases.
4. The synthetic and repetitive dataset probably makes it easy to learn the
   output format and frequent patterns without monotonically improving
   generalization.

## How to maximize useful learning

1. Use the 1,000-step checkpoint provisionally and apply early stopping based
   on accuracy or macro-F1 rather than validation loss alone.
2. Build a larger, independent, and balanced evaluation set. At minimum, use
   hundreds of real examples per class and separate sarcasm, negations, mixed
   sentiment, and ambiguous messages.
3. Deduplicate and diversify the training data. Replacing nearly identical
   synthetic variations with real messages and difficult cases will likely add
   more value than repeating another epoch.
4. Balance the classes and analyze macro-F1 and recall per class. The current
   evaluation set contains 40 negative cases but only 10 positive and 10
   neutral cases, so accuracy favors the negative class.
5. Test 8 and 16 LoRA layers, changing one variable at a time. There is enough
   memory headroom, but every result must be selected using independent data.
6. Test lower learning rates (`5e-6`, `2e-6`) and a scheduler with warmup and
   decay. Compare multiple random seeds because the small evaluation set can
   produce highly variable results.
7. Keep frequent checkpoints and do not assume that the final one is the best.

## Provisionally recommended adapter

`adapters-1epoch/0001000_adapters.safetensors`

To evaluate it:

```bash
python evaluate_sentiment.py \
  --adapter-path ./adapters-1epoch \
  --adapter-file ./adapters-1epoch/0001000_adapters.safetensors
```
