# Rules for Generating Amazon Reviews Examples

## Objective

Create a compact, high-quality dataset for training a small-context model that assists a procurement department. The buyer generally knows what product is needed, but the request may omit an essential definition; for example, it may request a masonry sealer without specifying the material to which it will be applied.

Each response must provide a useful, concise purchase recommendation supported exclusively by the buyer's requirements, catalog data, and supplied reviews. The initial domain is `Industrial_and_Scientific`, and all derived training content must be in English.

## 1. Dataset unit

- Generate **one comprehensive example per `parent_asin`**.
- Do not generate a separate example per review or per task.
- An example must jointly address:
  - the purchase recommendation;
  - supported advantages;
  - reported risks, limitations, and problems.
- Do not generate `use_cases` or suggest alternative uses. The objective is to evaluate the requested purchase, not broaden the product's purpose.
- All reviews, variants, and derived data for the same `parent_asin` must belong to a single split.
- The initial split is:
  - training: 4,000 products;
  - evaluation: 500 products;
  - test: 500 products.
- Assign splits by product before generating responses. No product, variant, review, or derived evidence from evaluation or test may appear in training.

## 2. Context selection

- Use no more than 40 reviews per product. This limit applies to input evidence, not training examples.
- Use fewer than 40 when additional reviews are redundant or exceed the context budget.
- The effective limit is the smaller of 40 reviews and the token budget defined for the target model.
- Prioritize relevant, informative, and diverse reviews that cover:
  - positive, intermediate, and negative ratings;
  - concrete benefits and failures;
  - material contradictions;
  - verified purchases and helpful votes, without treating either as a guarantee of truthfulness;
  - different variants or usage conditions when identified.
- Avoid duplicate, near-duplicate, empty, purely emotional, or product-uninformative reviews.
- Do not use strictly proportional sampling if it hides material minority evidence. Selection must preserve diversity without artificially exaggerating the frequency of positive or negative opinions.
- Review order must not imply priority or consensus.
- If recurrence is reported, its denominator must come from the complete set of relevant product reviews, not from a stratified sample that changed the proportions. If that denominator is unavailable, describe the reports without estimating frequency.

## 3. Example suitability

- **Reject** an example if more than 50% of the reviews do not refer to the product, or if the product is entirely outside the domain and has no reasonable industrial application.
- If no more than 50% of the reviews are unrelated, ignore them and use only the relevant evidence.
- Mark the example as **pending** if, after excluding unrelated evidence, it is impossible to create a supported response or determine whether the example is suitable.
- A correctable response, a negative review, contradictory evidence, or a do-not-buy recommendation is not grounds for rejection.
- An incomplete request is also not grounds for rejection: the model must identify the missing information and refrain from deciding when that information is essential.

## 4. Evidence rules

1. Use only the supplied catalog, buyer requirements, and reviews.
2. Always distinguish:
   - catalog claims;
   - user-reported experiences;
   - known buyer requirements;
   - information missing from the decision.
3. Do not invent specifications, certifications, compatibility, guarantees, causes, frequency, application conditions, or safety claims.
4. A seller claim is not equivalent to verified performance.
5. Describe an isolated report as isolated; never present it as a recurring problem.
6. A problem may be called **recurring** only if it appears in at least two independent reviews and exceeds 20% of the complete set of relevant reviews. If the complete set is unknown, do not use the label `recurring`.
7. Do not extrapolate percentages from the selected sample to all buyers.
8. Distinguish functional, compatibility, safety, logistics, and commercial risks when that distinction is relevant to the purchase.
9. A solution proposed by a user is not a validated procedure.
10. A high or low rating does not by itself demonstrate quality or defects.
11. If there are no concrete risks or problems, state that briefly; do not invent generic limitations.
12. Preserve contradictions in the evidence and explain what cannot be confirmed.
13. Do not assume that different variants share dimensions, materials, accessories, or performance.
14. Do not turn the absence of reports into evidence that problems do not exist.

An isolated, specific, and credible safety report may affect the recommendation, but it remains isolated and must not be called recurring.

## 5. Single response structure

All responses must use this compact contract:

```text
Purchase recommendation:
- Result: BUY | CONDITIONAL_BUY | DO_NOT_BUY | INSUFFICIENT_EVIDENCE
- Rationale: ...

Missing information:
- ...

Pros:
- ...

Risks and limitations:
- ...
```

Formatting rules:

- Emit exactly one `Result`.
- Keep `Rationale` concise and explain the primary reason for the decision.
- Include `Missing information` only when missing information materially affects the decision. Omit the entire section when it does not apply.
- `Pros` and `Risks and limitations` should normally contain two to four points each, but may contain fewer when evidence is limited.
- Do not repeat the same fact across sections.
- Include inherent limitations and isolated or recurring problems under `Risks and limitations`. Clearly identify which is which when the evidence supports that distinction.
- Do not create a separate `Common problems` section, because it usually duplicates risks and disadvantages.
- Do not create a `Use cases` section.
- If there is no concrete evidence for a required section, use a brief statement describing the evidence limitation instead of inventing content.

## 6. Purchase decisions

- **`BUY`**: The product meets known requirements, the evidence is sufficient, and no material conditions or risks remain unresolved.
- **`CONDITIONAL_BUY`**: Favorable evidence exists and the purchase depends on a concrete, verifiable, limited condition. State the condition specifically.
- **`DO_NOT_BUY`**: The evidence demonstrates incompatibility, failure to meet a mandatory requirement, a recurring material failure, an objective discrepancy, or a sufficiently credible safety risk.
- **`INSUFFICIENT_EVIDENCE`**: Information essential to the decision is missing, such as compatibility, certification, material, dimensions, application conditions, or a mandatory buyer requirement.

Precedence:

1. Failed mandatory requirement, demonstrated incompatibility, or material risk → `DO_NOT_BUY`.
2. Missing essential information or unresolved essential contradiction → `INSUFFICIENT_EVIDENCE`.
3. Favorable evidence with a verifiable, nonessential condition → `CONDITIONAL_BUY`.
4. Sufficient evidence with no material conditions → `BUY`.

Do not use `CONDITIONAL_BUY` when the missing information fully determines suitability or safety; use `INSUFFICIENT_EVIDENCE`.

Example: if a request calls for a masonry sealer but does not specify whether the surface is concrete, brick, stone, or painted material, and compatibility depends on the substrate, the result must be `INSUFFICIENT_EVIDENCE`, not `CONDITIONAL_BUY`.

## 7. Example quality

A high-quality example must:

- teach useful, generalizable behavior rather than memorize review wording;
- prioritize evidence material to the purchase decision;
- be concise and avoid exhaustive enumeration;
- preserve real uncertainty and contradictions;
- state conditions and missing information in concrete, verifiable terms;
- avoid promotional language, absolute claims, and generic recommendations;
- remain consistent across `Result`, `Rationale`, `Missing information`, `Pros`, and `Risks and limitations`.

Review redundant, trivial, or excessively similar examples even when they refer to different products. Dataset diversity should come from different product types, decisions, risks, constraints, and evidence patterns.

## 8. Final validation

Before approving a response, verify that:

- Exactly one example exists for the `parent_asin`.
- The product and all its evidence belong to a single split.
- The response follows the single structure and contains exactly one valid decision.
- The decision follows the precedence rules.
- Every material claim is supported.
- Catalog claims, reviews, requirements, and missing information are correctly separated.
- Isolated reports are not presented as recurring.
- Every recurrence claim uses a valid denominator rather than a disproportionate sample.
- Severity is proportional to the evidence.
- No specifications, compatibility, certifications, guarantees, causes, or application conditions were invented.
- No alternative uses were suggested and no `Use cases` section was added.
- Content is not repeated across sections.
- The response fits the target model's context and output budget.
- Only the `assistant` message was corrected; product data, reviews, input messages, ASINs, and metadata were never modified.
