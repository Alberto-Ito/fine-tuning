# Banking77 — consolidated test comparison

All results cover the same 3,080-example test split and 77 labels.

| Model / agent | Accuracy | Macro-F1 | Weighted-F1 | Input tokens | Output tokens | Total tokens | Request configuration |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B LoRA (2 epochs) | 0.8506 | 0.8509 | 0.8509 | 38,812 | 0 | 38,812 | local classifier |
| Qwen3.5-0.8B LoRA (2 epochs) | 0.8688 | 0.8689 | 0.8689 | 38,779 | 0 | 38,779 | local classifier |
| gpt-5.6-luna (direct) | 0.8519 | 0.8437 | 0.8437 | 1,343,982 | 24,798 | 1,368,780 | client prompt + labels |
| banking77-agent v2 | 0.7815 | 0.7739 | 0.7739 | 2,804,333 | 19,168 | 2,823,501 | user input only; server-side instructions |

## Main findings

- The persisted agent trails direct GPT-5.6 Luna by 7.05 percentage points in accuracy and 6.98 points in macro-F1.
- The agent uses 2.06x the API-reported tokens of the direct run because its longer persisted instructions are included server-side on every request.
- Qwen3.5-0.8B has the strongest result: 8.73 accuracy points and 9.50 macro-F1 points above the agent.
- The dedicated agent endpoint's validation response reported the underlying model field as `gpt-4o`, despite the experiment directory name. The evaluated serving target is therefore identified primarily by its immutable agent reference: `banking77-agent:2`.

## Agent run

- Completed predictions: 3080/3080
- Agent reference: `banking77-agent`, version `2`
- Client-supplied content: only the individual customer message as user input
- Outputs outside the 77-label set: 1
- Mean latency: 2.066 seconds
- Latency p50/p95: 1.719 / 3.025 seconds

## Token methodology

- Qwen token counts were calculated over all test texts using each local tokenizer. The classifiers return logits, so output tokens are zero.
- Direct GPT usage is reported by the Responses API and includes the client-supplied compact instructions and label list on every request.
- Agent usage is reported by the dedicated agent endpoint. The client sent no instructions, labels, or dataset context; input usage includes the persisted server-side agent instructions.
- Token figures cover test inference only and exclude Qwen fine-tuning.
