# Zava Retail Agent — SFT data expansion

This project expands the public Zava Retail Agent SFT dataset with grounded synthetic conversations. NeMo Curator and NeMo Data Designer run orchestration and validation on the Mac; all LLM inference is sent to a Microsoft Foundry deployment through its OpenAI-compatible v1 endpoint.

The official `sft_test.jsonl` is copied for reference and remains frozen. Generated examples are merged only into a new training file.

## Installed environment

```bash
cd /Users/albertoito/Code/fireworks/fine-tuning/Fireworks/ZavaRetailAgent
source .venv/bin/activate
python scripts/verify_install.py
```

Pinned top-level packages are in `requirements.txt`. The current environment uses Python 3.12 ARM64 and contains `nemo-curator[sdg_cpu]` plus Data Designer. It contains no CUDA or local inference server.

Important platform note: NeMo Curator 1.3.0 installs on this Mac but its package explicitly refuses to import outside Linux. The generation pipeline therefore invokes its Data Designer component directly on macOS. This provides the required synthetic-data workflow while Microsoft Foundry performs all model inference. Running NeMo Curator pipeline stages themselves later requires Linux (for example, an Ubuntu VM or container on a supported host); no unsupported platform patch is applied here.

## Microsoft Foundry setup

In Microsoft Foundry:

1. Open the Foundry resource/project that will pay for generation.
2. Deploy a chat-completions-capable model. For the first iteration, use a capable teacher such as GPT-4.1 mini and name the deployment `zava-datagen` (or use any name and place it in `.env`).
3. Confirm the deployment status is `Succeeded` and test it once in the playground.
4. Copy the resource's OpenAI-compatible v1 endpoint. It should end with `/openai/v1`, for example `https://RESOURCE.services.ai.azure.com/openai/v1`.
5. Copy one resource API key. Do not paste it into source code or chat.
6. Check that the deployment has enough tokens-per-minute quota. The project starts with two parallel requests to avoid throttling.

Create the local secret file:

```bash
cp .env.example .env
```

Fill in these three connection values inside `.env`:

```dotenv
FOUNDRY_BASE_URL=https://RESOURCE.services.ai.azure.com/openai/v1
FOUNDRY_MODEL_DEPLOYMENT=zava-datagen
FOUNDRY_API_KEY=YOUR_KEY
```

`.env` contains only the Foundry connection settings and is ignored by Git. Data Designer's local home is derived automatically from the project location, so moving the directory does not require configuration changes. Each run is defined by a separate JSON file under `config/`. The iteration ID and all output names are derived from the config filename rather than duplicated inside the JSON.

Verify connectivity without generating dataset rows:

```bash
python scripts/foundry_smoke_test.py
```

Expected output: `FOUNDRY_CONNECTION_OK`.

## Iteration 01 — five examples

Create five database-grounded scenario seeds:

```bash
python scripts/prepare_seeds.py --config config/iteration_01.json
```

Generate all examples defined by iteration 01 through Foundry:

```bash
python scripts/generate_sft.py --config config/iteration_01.json
```

Validate iteration 01:

```bash
python scripts/validate_sft.py --config config/iteration_01.json
```

## Iteration 02 — 500 examples

Run this only after iteration 01 passes generation, automatic validation, and manual review:

```bash
python scripts/prepare_seeds.py --config config/iteration_02.json
python scripts/generate_sft.py --config config/iteration_02.json
python scripts/validate_sft.py --config config/iteration_02.json
```

Both configurations execute the exact same `DataDesigner.create()` pipeline, artifact persistence, output conversion, and validation. Their target counts and output namespaces come exclusively from their config files.

Important outputs:

- `data/final/iteration_01_sft_train_extended.jsonl`: iteration 01 original train plus accepted synthetic examples.
- `data/final/iteration_02_sft_train_extended.jsonl`: iteration 02 equivalent, created only after that run.
- `data/final/iteration_XX_sft_generated_validation.jsonl`: validation slice drawn only from that iteration's generated examples.
- `reports/iteration_01_validation.json`: counts by intent and acceptance status.
- `reports/iteration_01_rejected.jsonl`: rejected records with reasons.

The validator enforces the chat schema, known tool names, one tool call at a time, matched tool results, JSON arguments, known database identifiers, intent-specific tool order, explicit confirmation before consequential actions, viable payment changes, and exact normalized deduplication against the original training set. A manual review is still required before submitting a fine-tuning job.

To retry only a failed seed without paying to regenerate accepted records, pass its ID to the same generator. The replacement is merged into the existing raw iteration output:

```bash
python scripts/generate_sft.py --config config/iteration_01.json --seed-id iter01-00003
```

Data Designer occasionally returns redundant assistant narration beside a tool call. The exporter canonicalizes such turns to the source Zava convention by retaining only the structured tool call; the validator independently rejects any mixed turn that remains.

## Data provenance

The files under `data/source/` and `reference/` come from Microsoft's public `microsoft-foundry/fine-tuning` repository, ZavaRetailAgent demo. Its upstream license is preserved at `reference/UPSTREAM_LICENSE`.
