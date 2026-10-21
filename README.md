# Small fine-tuned models vs. large general-purpose LLMs

This repository investigates a practical question:

> Can a small, task-specific model be fine-tuned to approach or exceed the
> quality of a much larger general-purpose LLM—such as GPT-4.1—while producing
> shorter outputs, lower latency, and a lower long-term inference cost?

The work is organized as an experimental comparison, not as a single model or
training framework. Each track builds a controlled task, fine-tunes a small
model, evaluates it against its base model and a large-LLM baseline, and records
quality, operational, and failure-mode metrics.

The current experiments use Qwen models from 0.6B to 4B parameters and compare
them with hosted general-purpose models including **GPT-4.1 mini**, GPT-5.6
Luna, and GPT-4o-based agents. GPT-4.1 mini is the measured GPT-4.1-family
baseline in the Amazon benchmark; this README does not attribute those results
to the full GPT-4.1 model.

## Project hypothesis

Large LLMs are strong zero-shot baselines, but they repeatedly pay the cost of
general reasoning, long instructions, and verbose generation. A small model can
instead absorb a stable task contract during fine-tuning.

The hypothesis is that, for narrow and repeatable workloads:

1. fine-tuning can close most or all of the quality gap with a large LLM;
2. specialization can improve format adherence and task-specific decisions;
3. shorter prompts and outputs reduce inference work on every request; and
4. the one-time data and training investment is amortized as request volume
   grows.

The repository is designed to test that hypothesis rather than assume it. A
small model is considered successful only when it is evaluated on held-out
data, against meaningful baselines, and with task-specific error analysis.

## Current evidence

### Amazon purchasing copilot: Qwen3-4B vs. GPT-4.1 mini

The current primary track fine-tunes Qwen3-4B to analyze product evidence and
produce grounded retail answers. It covers product pros and cons, use cases,
common problems, and purchase recommendations.

The first three-model benchmark used 200 held-out products, with 50 cases per
task. Responses from Qwen3-4B base, Qwen3-4B FT, and GPT-4.1 mini were judged
blindly by GPT-6 Sol.

| Model | Quality /100 | Pass rate | Critical-error rate | Mean latency | Output tokens |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 mini | **81.61** | **65.0%** | **17.0%** | 9.93 s | 69,066 |
| Qwen3-4B FT | 74.58 | 47.5% | 23.0% | **1.32 s** | **17,670** |
| Qwen3-4B base | 65.09 | 25.5% | 46.5% | 1.54 s | 57,978 |

This run shows the central trade-off clearly:

- GPT-4.1 mini delivered the best overall quality.
- Fine-tuning improved Qwen3-4B by **9.49 quality points**, increased pass rate
  by **22 percentage points**, and halved its critical-error rate relative to
  the base model.
- The fine-tuned model generated **74.4% fewer output tokens** than GPT-4.1
  mini and had much lower observed latency. Provider and deployment differences
  mean the latency figures are operational observations, not a pure architecture
  benchmark.
- On `purchase_recommendation`, Qwen3-4B FT reached a 76% pass rate, versus 58%
  for GPT-4.1 mini, although GPT-4.1 mini remained stronger globally.

The latest 103-case benchmark narrows the task to purchase decisions and uses a
newer fine-tune, `ft-qwen3-4b-amazon-v2`:

| Model | Judge score /100 | Decision accuracy | Macro-F1 | Pass rate | Critical-error rate | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B FT v2** | **59.24** | **45.63%** | **0.3136** | **30.10%** | **66.02%** | **72.82%** |
| GPT-4.1 mini | 39.48 | 31.07% | 0.2100 | 3.88% | 79.61% | 11.65% |
| Qwen3-4B base | 41.61 | 27.18% | 0.2696 | 0.97% | 80.58% | 15.53% |

In this narrower test, the fine-tuned 4B model beat GPT-4.1 mini on every
headline metric shown above. It is still **not production-ready**: a 66.02%
critical-error rate and weak minority-class performance make autonomous purchase
decisions unsafe. The result is evidence that specialization can outperform a
large baseline on a constrained contract, not evidence that the small model is
generally superior.

Detailed reports:

- [200-case three-model benchmark](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_10k/REPORT_THREE_MODELS_GPT6_JUDGE.md)
- [Qwen3-4B fine-tune vs. base A/B report](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_10k/REPORT.md)
- [103-case purchase-decision benchmark](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_v2/REPORT.md)

### Banking77: the initial proof of concept

Banking77 established the first controlled result on a closed 77-class intent
classification task. The same 3,080 test examples were used for the local
classifiers and hosted baselines.

| Model | Accuracy | Macro-F1 | Total test tokens |
|---|---:|---:|---:|
| **Qwen3.5-0.8B LoRA** | **86.88%** | **86.89%** | **38,779** |
| Qwen3-0.6B LoRA | 85.06% | 85.09% | 38,812 |
| GPT-5.6 Luna, direct | 85.19% | 84.37% | 1,368,780 |
| GPT-5.6 Luna agent | 84.71% | 83.99% | 15,101,827 |
| GPT-4o agent | 78.15% | 77.39% | 2,823,501 |

For this narrow classification task, the 0.8B fine-tuned model achieved the
best quality and the direct large-model baseline processed **35.3 times** more
tokens. The agent token totals include persisted instructions and are not
equivalent to billable cost; cached and uncached tokens must be priced
separately.

See [PROJECT_CONCLUSIONS.md](PROJECT_CONCLUSIONS.md) for the full analysis,
break-even model, limitations, and production recommendations.

## How the experiments are structured

The project follows the same comparison pattern across tasks:

1. **Define a narrow contract.** Specify the allowed tasks, labels or output
   schema, grounding rules, and unacceptable errors.
2. **Build traceable data.** Preserve source evidence, isolate products or
   entities across splits, deduplicate, validate, and record dataset versions.
3. **Establish baselines.** Evaluate the small base model and a capable hosted
   LLM, currently including GPT-4.1 mini.
4. **Fine-tune the small model.** Use LoRA or supervised fine-tuning and retain
   the model, configuration, and training artifacts.
5. **Evaluate held-out behavior.** Measure task quality, latency, token usage,
   output length, class balance, critical errors, and model-specific failure
   modes.
6. **Decide using the quality/cost frontier.** Prefer the smallest model that
   meets the required quality and safety threshold; route uncertain cases to a
   larger fallback model when necessary.

## Repository map

### Active work

- [`Fireworks/Amazon reviews 2023/`](Fireworks/Amazon%20reviews%202023/README.md):
  current end-to-end benchmark. It contains Amazon Reviews 2023 preparation,
  grounded SFT generation, data validation, Qwen3-4B fine-tunes, evaluations
  against GPT-4.1 mini, and judge reports.
- [`Fireworks/ZavaRetailAgent/`](Fireworks/ZavaRetailAgent/README.md): retail
  tool-use SFT data-generation and validation experiments.
- [`Hugging Face/`](Hugging%20Face/README.md): configurable local
  Transformers + PEFT/LoRA pipeline for training, evaluation, and prediction.
- [`Base Model/`](Base%20Model/README.md): zero-shot Qwen evaluations used to
  measure the gain produced by fine-tuning.
- `MS Foundry/`: hosted-model, agent, and deployment experiments, including
  GPT-based Banking77 baselines and Qwen deployment investigations.

### Historical and reference material

- `OLD_Qwen3-0.6B/` and `OLD_Qwen3.5-0.8B/`: original per-model Banking77
  implementations and two-epoch results.
- [`Legacy Mac Experiments/`](Legacy%20Mac%20Experiments/README.md): early
  Hugging Face and MLX experiments retained for provenance.
- [`PROJECT_CONCLUSIONS.md`](PROJECT_CONCLUSIONS.md): the original consolidated
  Banking77 conclusions and economic break-even framework.

For new local classification training, use the maintained
[`Hugging Face/`](Hugging%20Face/README.md) pipeline. For the current comparison
against GPT-4.1 mini, start with the Amazon Reviews track and its evaluation
reports.

## What the results do—and do not—show

The evidence supports the repository's working thesis: task specialization can
make a small model substantially better than its base version, much cheaper in
output volume, and competitive with a much larger LLM. In the latest constrained
purchase benchmark, the fine-tuned 4B model also outperformed GPT-4.1 mini.

The evidence does **not** show that small fine-tuned models replace large LLMs
for open-ended generation or general reasoning. Results depend on the dataset,
task contract, prompt, deployment, and evaluator. Automated LLM judges must be
calibrated with human review, and high critical-error rates override attractive
aggregate scores.

The intended production pattern is therefore often hybrid:

```text
request -> small specialized model -> confidence / rule checks
                                      |-> accepted response
                                      `-> large-LLM or human fallback
```

This keeps the recurring workload on the efficient model while preserving a
safer path for ambiguous, novel, or high-risk cases.

## Cost and amortization

Fine-tuning introduces a fixed cost `F`: data preparation, training,
evaluation, and deployment. Serving introduces a per-request cost:

```text
small-model cost = F + N × c_small
large-LLM cost   =     N × c_large

break-even requests = F / (c_large - c_small)
```

The experiments collect the token, latency, and throughput inputs needed for
this calculation. A financial claim should use the actual provider prices,
cached-token rates, hardware utilization, and engineering cost for the target
deployment. Raw token ratios alone are not a bill.

## Reproducibility and data handling

- Use held-out entities or products, not random rows, to prevent leakage.
- Keep model-visible content separate from provenance metadata.
- Version source data, transformations, prompts, model IDs, deployments, and
  judge configuration.
- Treat generated predictions and source reviews as potentially sensitive.
- Review model and dataset licenses separately from this repository's license.
- Do not promote an experiment to production based only on ROUGE, Token F1, or
  a single automated judge.

## License

The project code is released under the [MIT License](LICENSE). Models, hosted
services, and datasets retain their own licenses and terms; review their source
documentation before redistribution or production use.
