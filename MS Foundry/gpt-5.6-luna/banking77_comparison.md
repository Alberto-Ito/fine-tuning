# Banking77 test comparison

| Model | Accuracy | Macro-F1 | Weighted-F1 | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B LoRA (2 epochs) | 0.8506 | 0.8509 | 0.8509 | 38,812 | 0 | 38,812 |
| Qwen3.5-0.8B LoRA (2 epochs) | 0.8688 | 0.8689 | 0.8689 | 38,779 | 0 | 38,779 |
| gpt-5.6-luna | 0.8519 | 0.8437 | 0.8437 | 1,343,982 | 24,798 | 1,368,780 |

Foundry predictions: 3080/3080 completed.

Token methodology:

- Qwen input tokens were recalculated over all 3,080 test texts with each checkpoint's local tokenizer, special tokens enabled, truncation at 128, and no padding. These sequence-classification models return logits rather than generated tokens, so output tokens are zero.
- GPT-5.6 Luna values are the API-reported usage totals. Each independent request includes the classification instructions and the full list of 77 valid labels, in addition to the test text.
- These figures cover test inference only; they do not include Qwen fine-tuning tokens.
