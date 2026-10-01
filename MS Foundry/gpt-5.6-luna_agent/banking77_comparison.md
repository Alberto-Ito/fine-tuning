# Banking77 — consolidated test comparison

All results cover the same 3,080-example test split and 77 labels.

| Model / agent | Accuracy | Macro-F1 | Weighted-F1 | Input tokens | Cached input | Output tokens | Total tokens | Request configuration |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B LoRA (2 epochs) | 0.8506 | 0.8509 | 0.8509 | 38,812 | 0 | 0 | 38,812 | local classifier |
| Qwen3.5-0.8B LoRA (2 epochs) | 0.8688 | 0.8689 | 0.8689 | 38,779 | 0 | 0 | 38,779 | local classifier |
| gpt-5.6-luna (direct) | 0.8519 | 0.8437 | 0.8437 | 1,343,982 | 0 | 24,798 | 1,368,780 | client prompt + labels |
| gpt-4o agent v2 | 0.7815 | 0.7739 | 0.7739 | 2,804,333 | 0 | 19,168 | 2,823,501 | user input only; server-side instructions |
| gpt-5.6-luna agent v3 | 0.8471 | 0.8399 | 0.8399 | 15,074,622 | 14,574,364 | 27,205 | 15,101,827 | user input only; server-side instructions |

## Main findings

- GPT-5.6 Luna agent v3 differs from direct GPT-5.6 Luna by -0.49 percentage points in accuracy and -0.37 points in macro-F1.
- GPT-5.6 Luna agent v3 differs from GPT-4o agent v2 by +6.56 accuracy points and +6.60 macro-F1 points.
- The agent uses 11.03x the API-reported tokens of the direct run because its longer persisted instructions are included server-side on every request.
- 96.7% of the v3 agent's input tokens were reported as cached input tokens.
- Qwen3.5-0.8B has the strongest result: 2.18 accuracy points and 2.90 macro-F1 points above the agent.
- The dedicated endpoint validation confirmed model `gpt-5.6-luna` and immutable agent reference `banking77-agent:3`.

## Agent run

- Completed predictions: 3080/3080
- Agent reference: `banking77-agent`, version `3`
- Reported model: `gpt-5.6-luna`
- Client-supplied content: only the individual customer message as user input
- Outputs outside the 77-label set: 1
- Mean latency: 2.379 seconds
- Latency p50/p95: 2.111 / 3.540 seconds

## Token methodology

- Qwen token counts were calculated over all test texts using each local tokenizer. The classifiers return logits, so output tokens are zero.
- Direct GPT usage is reported by the Responses API and includes the client-supplied compact instructions and label list on every request.
- Agent usage is reported by the dedicated agent endpoint. The client sent no instructions, labels, or dataset context; input usage includes the persisted server-side agent instructions.
- Token figures cover test inference only and exclude Qwen fine-tuning.
