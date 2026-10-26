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
them with hosted general-purpose models including **GPT-4.1 mini**,
**GPT-5**, GPT-5.6 Luna, and GPT-4o-based agents. GPT-4.1 mini is the measured GPT-4.1-family
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

### Amazon purchasing copilot: Qwen3-4B vs. GPT-4.1 mini and GPT-5

The current primary track fine-tunes Qwen3-4B on Fireworks AI to analyze product
evidence (catalog data plus customer reviews) and produce grounded purchase
recommendations. Every response must emit exactly one decision:
`BUY`, `CONDITIONAL_BUY`, `DO_NOT_BUY`, or `INSUFFICIENT_EVIDENCE`, followed by a
rationale, missing information, pros, and risks.

Two benchmarks have been run. Both use held-out products and GPT-6 Sol as a
blind judge (model identities hidden and order randomized).

#### 103-case purchase-decision benchmark (latest, October 20–22, 2026)

Fine-tune: `ft-qwen3-4b-amazon-v2`, evaluated with an explicit output prompt.
Reference distribution: 51 `DO_NOT_BUY`, 24 `CONDITIONAL_BUY`,
24 `INSUFFICIENT_EVIDENCE`, and 4 `BUY`. All models returned 103/103 valid,
correctly formatted answers.

**Decision quality against the reference labels** (deterministic, primary metric):

| Model | Accuracy | Macro-F1 | Weighted-F1 | `BUY` predictions (ref. 4) |
|---|---:|---:|---:|---:|
| **Qwen3-4B FT v2** | **54.37%** | **0.4607** | **0.5423** | 7 |
| GPT-4.1 mini | 43.69% | 0.3749 | 0.3959 | 12 |
| GPT-5 | 37.86% | 0.3330 | 0.4529 | 52 |
| Qwen3-4B base | 27.18% | 0.2333 | 0.1689 | 6 |

**Explanation quality according to the blind GPT-6 Sol judge.** The judge was
run twice, once per comparison group, so scores should only be compared within
the same round:

| Judge round | Model | Judge score /100 | Pass rate | Critical-error rate | Preferred |
|---|---|---:|---:|---:|---:|
| FT v2 vs. GPT-4.1 mini vs. base | **Qwen3-4B FT v2** | **61.00** | **34.95%** | **59.22%** | **64.08%** |
| | GPT-4.1 mini | 53.64 | 10.68% | 69.90% | 29.13% |
| | Qwen3-4B base | 42.75 | 1.94% | 81.55% | 6.80% |
| FT v2 vs. GPT-5 vs. base | **GPT-5** | **63.70** | **35.92%** | 62.14% | **58.25%** |
| | Qwen3-4B FT v2 | 60.40 | 33.01% | **58.25%** | 33.98% |
| | Qwen3-4B base | 43.12 | 2.91% | 79.61% | 7.77% |

Takeaways:

- The fine-tune doubles the base model's decision accuracy and makes the most
  accurate purchase decisions of every model tested, including GPT-5
  (+16.51 pp accuracy, +0.128 macro-F1).
- GPT-5 writes the best explanations (higher grounding, evidence coverage, and
  judge preference) but is strongly biased toward `BUY`: it chose it in 52 of
  103 cases, versus 4 in the reference.
- The fine-tune beats GPT-4.1 mini on both decision accuracy and judge score.
- **Nothing here is production-ready.** Even the best models (FT v2 and GPT-5)
  had critical errors in 58–62% of cases. A reasonable design is the fine-tune for the
  decision, a generalist such as GPT-5 to review grounding or draft the
  justification, and human review for high-risk cases.

#### 200-case four-task benchmark (October 15, 2026)

Fine-tune: `ft-qwen3-4b-amazon-10k`. Tasks: pros and cons, use cases, common
problems, and purchase recommendation (50 cases each).

| Model | Judge score /100 | Pass rate | Critical-error rate | Output tokens | Mean latency |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 mini | **81.61** | **65.0%** | **17.0%** | 69,066 | 9.93 s |
| Qwen3-4B FT | 74.58 | 47.5% | 23.0% | **17,670** | **1.32 s** |
| Qwen3-4B base | 65.09 | 25.5% | 46.5% | 57,978 | 1.54 s |

On this broader task mix, GPT-4.1 mini led overall quality. Fine-tuning still
added 9.49 judge points over the base model, produced 74% fewer output tokens
than GPT-4.1 mini, and led the purchase-recommendation task on pass rate
(76% vs. 58%). Latencies come from different providers and deployments.

The evidence shows that specialization can beat large baselines on a
constrained decision contract. It does not show that the small model is better
in general. GPT-6 Sol is a single automated judge and must still be calibrated
with blind human review.

Detailed reports:

- [103-case benchmark: FT v2 vs. GPT-4.1 mini vs. base](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_v2/STATUS.md)
- [103-case benchmark: FT v2 vs. GPT-5 vs. base](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_v2/STATUS_GPT5.md)
- [200-case three-model benchmark](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_10k/REPORT_THREE_MODELS_GPT6_JUDGE.md)
- [Qwen3-4B fine-tune vs. base A/B report](Fireworks/Amazon%20reviews%202023/evaluation/ft_qwen3_4b_amazon_10k/REPORT.md)

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
   LLM, currently including GPT-4.1 mini and GPT-5.
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
  against GPT-4.1 mini and GPT-5, and judge reports.
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
against GPT-4.1 mini and GPT-5, start with the Amazon Reviews track and its
evaluation reports.

## What the results do—and do not—show

The evidence supports the repository's working thesis: task specialization can
make a small model substantially better than its base version, much cheaper in
output volume, and competitive with a much larger LLM. In the latest constrained
purchase benchmark, the fine-tuned 4B model made more accurate purchase
decisions than both GPT-4.1 mini and GPT-5, while GPT-5 still produced the
better-written explanations.

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
