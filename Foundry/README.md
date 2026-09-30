# Banking77 evaluation on Microsoft Foundry

The evaluation sends every example from the official Banking77 test split to
the configured Responses API deployment. Results are appended to disk as they
arrive, so rerunning the command resumes missing cases.

```bash
cd fine-tuning
export FOUNDRY_API_KEY='...'
OLD_Qwen3.5-0.8B/.venv/bin/python Foundry/run_banking77.py
OLD_Qwen3.5-0.8B/.venv/bin/python Foundry/compute_metrics.py
```

The API key is read only from the environment and must not be committed.
The runnable Azure deployment name is `gpt-5.6-luna` (the `foundry/` prefix is
the provider-qualified model name, not part of the deployment name).
