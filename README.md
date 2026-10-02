# Configurable fine-tuning

This project trains, evaluates, and runs LoRA sequence-classification adapters
from YAML configuration files. The maintained implementation lives in
`src/finetuning/`; model and dataset choices are configuration-driven, so the
same three entry points can be used for Qwen3, Qwen3.5, and compatible future
models and classification datasets.

The checked-in examples use the 77-class Banking77 intent-classification
dataset. This repository does not provide a general text-generation or
instruction-tuning pipeline.

## Repository layout

```text
configs/datasets/       Dataset sources, columns, splits, and sampling
configs/experiments/    Model, LoRA, training, and output settings
src/finetuning/         Canonical train, evaluate, and predict package
outputs/                Checkpoints, final adapters, metrics, and predictions
reports/                Confusion matrices and result tables
FIRST_mac-local/        Historical first experiments; reference only
OLD_*/                  Historical per-model implementations; reference only
MS Foundry/             Separate Azure AI Foundry experiments
```

New local training work should use `src/finetuning` and the configurations
under `configs/`. The historical directories are not part of the supported
command-line workflow described below.

## Requirements and installation

- Python 3.10 or newer.
- Internet access for the first download of Hugging Face models and datasets,
  unless all required files are already cached or local.
- Enough memory for the selected base model, optimizer state, activations, and
  evaluation batches. There is no single minimum: memory usage depends on the
  model, sequence length, batch sizes, and LoRA targets.
- A CUDA GPU or Apple Silicon with MPS is recommended for training. CPU-only
  execution may be possible but is not a practical default for the included
  experiments.

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The model loader uses the configured PyTorch dtype, which defaults to
`bfloat16`. Verify that the selected device supports it; otherwise set `dtype`
to a compatible PyTorch dtype name such as `float32`. Training enables the MPS
fallback environment setting when it has not already been set.

`requirements.txt` currently installs Transformers from its GitHub `main`
branch. For a reproducible long-lived experiment, record the exact resolved
commit and the output of `pip freeze` alongside the run.

## Quick start

Run a small pilot before a full experiment:

```bash
finetune-train --config configs/experiments/qwen3_06b_banking77_pilot100.yaml
finetune-train --config configs/experiments/qwen35_4b_v1_banking77_pilot100.yaml
```

The Qwen3.5-4B v1 configuration is a local sequence-classification pilot. Read
[`MS Foundry/qwen35_4b_v1/README.md`](MS%20Foundry/qwen35_4b_v1/README.md)
before planning a Foundry upload: Microsoft currently excludes this model from
LoRA adapter deployment, and the repository's classification adapter is not a
generative vLLM adapter.

Run the full included configurations:

```bash
finetune-train --config configs/experiments/qwen3_06b_banking77.yaml
finetune-train --config configs/experiments/qwen35_08b_banking77.yaml
```

The same commands are available as modules when the package has not been
installed in editable mode:

```bash
PYTHONPATH=src python -m finetuning.train \
  --config configs/experiments/qwen35_08b_banking77.yaml

PYTHONPATH=src python -m finetuning.evaluate \
  --config configs/experiments/qwen35_08b_banking77.yaml \
  --adapter outputs/banking77/qwen35_08b/model

PYTHONPATH=src python -m finetuning.predict \
  --model outputs/banking77/qwen35_08b/model \
  --text "Why was my transfer declined?"
```

Paths passed on the command line and output paths in experiment files are
resolved relative to this project directory unless they are absolute.

## Commands

### Train

```text
finetune-train --config EXPERIMENT_YAML
```

`--config` is required. Training loads the referenced dataset configuration,
creates a validation split from the configured training split, applies the
optional example limit, tokenizes all three splits, trains the adapter, selects
the best checkpoint by validation macro F1, evaluates it, and writes the final
model and run artifacts.

### Evaluate

```text
finetune-evaluate --config EXPERIMENT_YAML --adapter ADAPTER_DIRECTORY
```

Both arguments are required. The experiment supplies the base model,
tokenization length, evaluation batch size, and dataset configuration. The
adapter directory must contain a PEFT adapter compatible with that base model.
The command evaluates the configured test split and prints a JSON object with
test metrics to stdout. It does not write the full training artifact set.

### Predict

Provide one or more texts directly:

```bash
finetune-predict \
  --model outputs/banking77/qwen35_08b/model \
  --text "Why was my transfer declined?" "Where is my card?"
```

Or read newline-delimited JSON:

```bash
finetune-predict \
  --model outputs/banking77/qwen35_08b/model \
  --input requests.jsonl \
  --output predictions.json
```

`--model` is required and must point to the saved adapter directory. Each input
line is expected to be a JSON object with a string `text` field:

```json
{"text": "Why was my transfer declined?"}
```

Without `--output`, the command prints a JSON array to stdout. With `--output`,
it writes the array to the specified file. Each item currently has this shape:

```json
{
  "text": "Why was my transfer declined?",
  "label_id": 42
}
```

Provide either `--text` or `--input`. If both are supplied, the current
implementation uses `--text`; if neither is supplied, it exits with an error.
Prediction currently returns numeric label IDs rather than label names.

## Experiment configuration

An experiment YAML controls the model, training schedule, LoRA adapter, and
artifact destinations. All included experiment configurations define the
following fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `run_name` | yes | Name stored in Trainer state and the run summary. |
| `model_name` | yes | Hugging Face model ID or local base-model path. |
| `dataset_config` | yes | Dataset YAML path. It may be project-relative or relative to the experiment file. |
| `max_steps` | yes | Positive value caps training by optimizer steps; `-1` uses epochs. |
| `num_train_epochs` | yes | Number of epochs used when `max_steps` is not positive. |
| `per_device_train_batch_size` | yes | Training batch size per device. |
| `per_device_eval_batch_size` | yes | Evaluation batch size per device. |
| `gradient_accumulation_steps` | yes | Mini-batches accumulated before an optimizer step. |
| `max_length` | yes | Tokenized sequence length; shorter examples are padded to this size. |
| `learning_rate` | yes | Initial optimizer learning rate. |
| `weight_decay` | yes | AdamW weight decay. |
| `warmup_ratio` | no | Fraction of planned steps used for warmup; defaults to `0`. |
| `lr_scheduler_type` | no | Transformers scheduler name; defaults to `cosine`. |
| `lora_rank` | yes* | LoRA rank in the flat configuration form. |
| `lora_alpha` | yes* | LoRA scaling value in the flat configuration form. |
| `lora_dropout` | yes* | LoRA dropout in the flat configuration form. |
| `lora_target_modules` | yes* | Model module names that receive LoRA adapters. |
| `logging_steps` | yes | Interval for Trainer log records. |
| `eval_steps` | yes | Validation interval. |
| `save_steps` | yes | Checkpoint interval. |
| `save_total_limit` | yes | Maximum number of Trainer checkpoints retained. |
| `seed` | no | Training and data seed; defaults to `42`. |
| `dtype` | no | Attribute name from `torch`, such as `bfloat16`; defaults to `bfloat16`. |
| `resume_from_checkpoint` | no | Checkpoint directory from which Trainer resumes. |
| `output_dir` | no | Trainer checkpoint directory; defaults to `outputs/checkpoints`. |
| `final_model_dir` | no | Final adapter and tokenizer directory; defaults to `outputs/model`. |
| `logging_dir` | no | Reserved log path; defaults to `outputs/logs`. |
| `metrics_dir` | no | Metrics and resource records; defaults to `outputs/metrics`. |
| `predictions_dir` | no | Per-example test predictions; defaults to `outputs/predictions`. |
| `report_dir` | no | Confusion-matrix destination; defaults to `reports`. |
| `confusion_matrix_filename` | no | Confusion-matrix filename; defaults to `confusion_matrix.csv`. |

`*` Instead of the four flat LoRA fields, an experiment may define a `lora`
mapping containing `enabled`, `rank`, `alpha`, `dropout`, `target_modules`, and
optionally `modules_to_save`. `enabled` defaults to `true`, and
`modules_to_save` defaults to `[score]`.

When `max_steps` is positive, Transformers treats it as the training limit and
it overrides the epoch-derived duration. `num_train_epochs` is still required
by the current configuration loader and is also used when estimating planned
steps for warmup if `max_steps` is not positive.

## Dataset configuration

A dataset YAML describes either a Hugging Face dataset or CSV files.

Common fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | no | Human-readable dataset name. |
| `source_type` | no | `huggingface` or `csv`; defaults to `csv`. The legacy key `loader` is also accepted. |
| `text_column` | yes | Source column containing input text. |
| `label_column` | yes | Categorical target column. |
| `train_split` | yes | Split from which training and validation data are made. |
| `test_split` | yes | Split used for final evaluation. |
| `validation_fraction` | yes | Fraction of the training split reserved for validation. |
| `seed` | no | Split and sampling seed; defaults to `42`. |
| `max_examples` | no | Maximum selected independently from each train, validation, and test split after shuffling. |

For a Hugging Face source:

```yaml
source_type: huggingface
huggingface_id: PolyAI/banking77
subset: optional-subset-name
trust_remote_code: false
```

`huggingface_id` is required; `subset` and `trust_remote_code` are optional.
Enabling `trust_remote_code` permits code supplied by the dataset repository to
run locally. Only enable it for a source you trust and review.

For CSV sources:

```yaml
source_type: csv
data_files:
  train: path-or-url-to-train.csv
  test: path-or-url-to-test.csv
source_label_column: category
label_column: label
```

`data_files` is passed to the Hugging Face CSV loader. Local relative paths are
interpreted by that loader from the process working directory, so project-root
paths or absolute paths are safest. `source_label_column` is optional; when it
differs from `label_column`, the source column is renamed. CSV labels are then
class-encoded.

The training split must support stratification by the label column. Label names
and numeric IDs come from the dataset feature metadata and must remain
consistent between training, evaluation, and the saved adapter.

## Resume a run

Set `resume_from_checkpoint` in the experiment YAML to a Trainer checkpoint,
for example:

```yaml
resume_from_checkpoint: outputs/banking77/qwen35_08b/checkpoints/checkpoint-2252
max_steps: 3378
```

This is a total-step target, not “3378 additional steps.” Keep the base model,
dataset label mapping, LoRA targets, optimizer-related settings, and output
layout compatible with the original run. A checkpoint directory is different
from `final_model_dir`: checkpoints include Trainer and optimizer state needed
to continue training, while the final model directory is intended for adapter
loading and prediction.

## Generated artifacts

After a successful training run, the configured destinations contain:

| Artifact | Contents |
| --- | --- |
| `output_dir/checkpoint-*` | Periodic Trainer checkpoints and resumable state. |
| `final_model_dir/` | Final PEFT adapter, adapter configuration, tokenizer, and training arguments. |
| `metrics_dir/run_summary.json` | Run identity, example counts, elapsed training time, global steps, and train/validation/test metrics. |
| `metrics_dir/test_metrics.json` | Accuracy, macro precision, macro recall, macro F1, and test loss. |
| `metrics_dir/test_metrics_per_class.csv` | Per-label precision, recall, F1, and support. |
| `metrics_dir/trainer_log_history.json` | Raw Trainer logging history. |
| `metrics_dir/resources.jsonl` | Timestamped process RSS and, on MPS, allocated/driver memory, plus Trainer log values. |
| `predictions_dir/test_predictions.jsonl` | Text, expected and predicted IDs/names, confidence, and correctness for every test example. |
| `report_dir/confusion_matrix.csv` | Rows are expected labels and columns are predicted labels. |

Accuracy is the fraction of correct predictions. Macro precision, recall, and
F1 calculate the metric separately for every class and average the class
values equally, which prevents large classes from dominating the headline
score.

Treat prediction files as potentially sensitive: they contain the original
input text. Output files and checkpoints may also be large. Decide explicitly
which artifacts should be committed or shared.

## Reproducibility notes

The configured seed is passed to Transformers and is used for the stratified
validation split and optional example sampling. Exact reproducibility is not
guaranteed across different devices, PyTorch/Transformers versions, kernels,
or distributed configurations. For a repeatable run, preserve:

- the experiment and dataset YAML files;
- the exact dependency versions and Transformers commit;
- base-model and dataset revisions or local snapshots;
- device type and relevant runtime settings;
- the checkpoint used when resuming.

The included remote CSV configurations and Hugging Face model/dataset IDs do
not pin content revisions. Cache or revision-pin external inputs when exact
reproduction matters.

## License

The project code is released under the [MIT License](LICENSE). Models and
datasets retain their own licenses and terms; review the relevant Hugging Face
model card and the Banking77 source before redistribution or production use.
