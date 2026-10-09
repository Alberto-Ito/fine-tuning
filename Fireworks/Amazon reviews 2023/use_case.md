# Industrial demo use-case ranking for fine-tuning an 8B model

## Objective

Select a practical subset of Amazon Reviews 2023 for an industry-oriented fine-tuning demo, then compare an 8B base model, the same 8B model after fine-tuning, and a larger Microsoft Foundry baseline such as `gpt-5.6-sol`.

The goal is not to claim that an 8B model is universally better. The demo should measure whether specialization provides sufficient quality with better cost, latency, privacy, deployment control, or output consistency.

## Ranking criteria

- Industrial relevance and business clarity.
- Availability of grounded evidence in the dataset.
- Ability to define repeatable SFT tasks.
- Objective or auditable evaluation.
- Clear differentiation between base and fine-tuned behavior.
- Fair comparison with a larger general-purpose model.
- Safety and hallucination risk.
- Demo clarity for a non-research audience.

## Executive ranking

| Rank | Use case | Recommended source | Why it is useful |
|---:|---|---|---|
| 1 | Technical purchasing and MRO compatibility copilot | `Industrial_and_Scientific` | Strong business narrative, rich attributes, and measurable grounded behavior. |
| 2 | Defect, complaint, and corrective-action classification | `Industrial_and_Scientific`, `Automotive`, tools | Structured outputs and deterministic metrics. |
| 3 | Voice-of-customer aspect and recurring-problem extraction | Industrial, tools, electronics | Connects reviews with quality and product analytics. |
| 4 | Grounded product FAQ assistant | Industrial and tools | Useful, but performance depends heavily on retrieval quality. |
| 5 | Personalized product recommendation | Multiple categories | Valuable, but primarily a ranking problem and difficult to compare fairly with a chat model. |

## 1. Technical purchasing and MRO compatibility copilot

### Proposal

Given a purchasing request and a small catalog of products, the assistant should:

1. Extract explicit requirements and constraints.
2. Identify missing critical specifications.
3. Ask targeted clarification questions.
4. Compare only evidence-supported properties.
5. Recommend suitable candidates or abstain.
6. State risks, uncertainties, and required verification.

Example request:

> I need a DC solid-state relay for a 24 V control system, approximately 15 A continuous load, limited enclosure ventilation, and frequent switching. Which candidate is appropriate, and what should I verify first?

### Why it ranks first

- It resembles a real procurement workflow.
- Product metadata supplies specifications and catalog evidence.
- Reviews provide field observations, failure modes, and compatibility warnings.
- The task tests extraction, clarification, grounded comparison, and safe abstention.
- Fine-tuning can teach stable domain behavior rather than transient catalog facts.

### What the 8B model should learn

- Requirement extraction into a stable schema.
- Distinction between required, preferred, and unknown attributes.
- Compatibility reasoning based only on supplied evidence.
- Clarification when critical information is missing.
- Evidence attribution and calibrated uncertainty.
- Refusal to infer safety-critical specifications from weak reviews.
- Consistent structured output.

### What it should not memorize

- Current price or availability.
- Product-specific facts that should come from catalog retrieval.
- Unsupported safety guarantees.
- ASIN identifiers as semantic knowledge.
- A simplistic rule that high rating means technical suitability.

### Recommended source subset

The curated layer currently contains 150,000 reviews and 14,000 products:

| Component | Target |
|---|---:|
| Core reviews | 120,000 |
| Long-tail reviews | 30,000 |
| Core products | 4,000 |
| Long-tail products | 10,000 |
| Initial high-quality SFT examples | 30,000–60,000 |

Do not force 100,000 SFT conversations merely because 100,000 reviews are available. One product can yield several tasks, but low-value repetition should be rejected.

### Suggested SFT task mix

| Task | Initial share |
|---|---:|
| Requirement extraction | 20% |
| Clarifying questions | 15% |
| Grounded product comparison | 20% |
| Conditional purchase recommendation | 15% |
| Pros, cons, and recurring issues | 10% |
| Structured compatibility decision | 10% |
| Abstention for insufficient evidence | 10% |

### Example output schema

```json
{
  "requirements": {
    "control_voltage": "24 V DC",
    "continuous_current": "15 A",
    "switching_frequency": "frequent",
    "cooling": "limited"
  },
  "missing_information": ["load type", "ambient temperature"],
  "decision": "insufficient_information",
  "candidate_assessment": [],
  "risks": ["thermal derating must be verified"],
  "confidence": "low"
}
```

### Primary metrics

- Requirement extraction precision, recall, and F1.
- Schema validity.
- Compatibility decision accuracy.
- Critical missing-information recall.
- Unsupported-claim rate.
- Evidence-entailment score.
- Appropriate abstention rate.
- Pairwise human preference.
- Latency, output tokens, throughput, and cost.

## 2. Defect, complaint, and corrective-action classification

Classify reviews into a controlled taxonomy such as damaged shipment, missing component, early failure, incorrect dimensions, misleading specification, incompatibility, poor documentation, overheating, leakage, or insufficient information.

Advantages:

- Mostly deterministic evaluation.
- Fine-tuning can improve stable labels and JSON output.
- Easy integration into quality or support triage.

Risks:

- Labels do not exist in the source and require annotation or teacher generation.
- Rating is not a substitute for severity.
- Corrective action depends on business policy that the dataset does not contain.

Start with 12–20 defect classes, include `other`, `insufficient_information`, and multilabel cases, and manually validate at least 500 test items.

## 3. Voice of Customer

Extract mentioned aspects and polarity, then aggregate recurring problems across products or subcategories. Candidate aspects include durability, fit, installation, accuracy, packaging, documentation, compatibility, noise, thermal behavior, and value.

This is useful for quality dashboards and supply-chain feedback. A fine-tuned 8B model may provide higher throughput and schema stability, but a frontier model with a strong prompt will also be competitive. The demo must therefore include cost, latency, and output-consistency measurements.

## 4. Grounded product FAQ assistant

Answer questions about a product using the same retrieved metadata and reviews for every compared model. Retrieval quality must be held constant; otherwise the experiment measures retrieval rather than fine-tuning.

Evaluate answer correctness, evidence entailment, unsupported claims, abstention, pairwise preference, and response length.

## 5. Personalized recommendation

Personalized recommendation is not the preferred first demo because Amazon Reviews does not expose every product impression or rejected candidate. It is primarily a ranking problem, and offline ranking metrics do not necessarily measure conversational quality. It can become a later phase with a specialized recommender and an 8B conversational layer.

## Recommended demo

Build a **technical MRO purchasing copilot with requirement extraction, clarification, grounded comparison, and safe abstention**. Add defect classification as a secondary workflow using the same category.

```text
Purchasing request
    -> extract requirements
    -> detect missing critical data
    -> ask clarifying questions
    -> compare candidates with evidence
    -> recommend or abstain
    -> produce structured output
```

## Experimental design

### Models

| Model | Purpose |
|---|---|
| 8B base, zero-shot | Establish the unadapted baseline. |
| 8B base, optimized prompt/few-shot | Measure prompting alone. |
| 8B with LoRA/QLoRA | Measure the contribution of fine-tuning. |
| `gpt-5.6-sol` in Microsoft Foundry | Larger general-purpose reference. |
| Optional intermediate Foundry model | Build a quality/cost curve. |

Create evaluation sets before investing in large-scale fine-tuning. Teacher-generated answers must pass filters, graders, and human review before training.

### Fair-comparison requirements

- Exactly the same test cases for every model.
- No test product, variant, review, or derived evidence in training.
- Identical retrieved context and token limits.
- Equivalent system policy, adjusted only for API requirements.
- Multiple runs for nondeterministic tasks.
- Blind human evaluation.
- Controlled output length to reduce verbosity bias.
- Report quality, latency, tokens, throughput, and cost together.

### Splits

1. Filter and deduplicate source data.
2. Group by `parent_asin`.
3. Assign products to train, validation, and test.
4. Reserve temporal and unseen-product slices inside test.
5. Keep every synthetic derivative of the same evidence in one split.

| Split | Product share | Purpose |
|---|---:|---|
| Train | 80% | Fine-tuning. |
| Validation | 10% | Hyperparameters and checkpoint selection. |
| Test IID | 5% | Similar tasks on unseen products. |
| Test temporal/OOD | 5% | Later or shifted products/subcategories. |

### Minimum viable test

Target approximately 5,000 cases across requirement extraction, clarification, product comparison, safe abstention, defect classification, adversarial prompts, long context, and unseen products.

## Demo scorecard

| Dimension | Metric | Initial target for fine-tuned 8B |
|---|---|---:|
| Format | Valid JSON/schema | >= 99% |
| Grounding | Evidence-supported claims | >= 95% |
| Safety | Critical missing-data recall | >= 95% |
| Reliability | Unsupported-claim rate | <= 3% |
| Task quality | Macro-F1 or accuracy | Report by slice |
| Human quality | Pairwise win/tie/loss | Report |
| Efficiency | P50/P95 latency | Report |
| Efficiency | Cost per 1,000 cases | Report |

The business success criterion is not necessarily beating the frontier model on every dimension. An 8B model can be the better deployment choice if it reaches the required quality threshold with materially better economics or control.

## Implementation phases

### Phase 0 — Evaluation first

- Freeze the task taxonomy and schemas.
- Create 300–500 manual gold cases.
- Run base and frontier baselines.
- Identify systematic, teachable errors.

### Phase 1 — Pilot dataset

- Filter 5,000–10,000 high-quality examples.
- Generate teacher outputs.
- Apply automatic grading and human review.
- Fine-tune and compare against baselines.

### Phase 2 — Main dataset

- Scale to 30,000–60,000 SFT examples.
- Add difficult negative and abstention cases.
- Re-run contamination, balance, and slice checks.

### Phase 3 — Foundry demo

- Deploy comparable endpoints.
- Use one inference and evaluation runner.
- Capture latency, usage, and cost.
- Run the complete evaluation and blind human review.

### Phase 4 — Decision

Compare frontier-only, fine-tuned-8B-only, and hybrid routing. Hybrid routing is likely the most realistic industrial outcome: the 8B handles routine extraction and classification, while a frontier model handles complex reasoning.

## Go/no-go criteria

Continue if few-shot prompting misses targets, errors are systematic and teachable, output schemas are stable, enough reviewed data exists, and the 8B provides meaningful cost or latency improvements.

Stop or redesign if the task mainly depends on dynamic knowledge, RAG and prompting solve it equally well, synthetic labels lack human agreement, the test is contaminated, gains only appear under an unfair frontier prompt, or safety requires authoritative external validation.

## Recommended decision

Proceed with the technical MRO purchasing pilot using `Industrial_and_Scientific`, an English-only derived dataset, and a gold test created before the main training set. The demo narrative should emphasize **specialization, control, grounding, and efficiency**, not universal model superiority.
