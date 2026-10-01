# Banking77 evaluation through the persisted Foundry agent

This evaluation invokes `banking77-agent` version 2 through its dedicated
Responses endpoint. Every request contains only the Banking77 test text as a
user message. No client-side classification instructions, label list, dataset
name, model name, or agent reference are added to the request body.

```bash
cd fine-tuning
export FOUNDRY_API_KEY='...'
OLD_Qwen3.5-0.8B/.venv/bin/python 'MS Foundry/gpt-5.6-luna_agent/run_banking77_agent.py'
OLD_Qwen3.5-0.8B/.venv/bin/python 'MS Foundry/gpt-5.6-luna_agent/compute_metrics.py'
```

The runner appends results incrementally and resumes missing indices when run
again. The API key is read from the environment and is never stored in output.
