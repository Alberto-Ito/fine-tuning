# Banking77 fine-tuning

Repository of Banking77 banking-intent classification experiments across local
Hugging Face training, base-model evaluation, Fireworks, and Microsoft Foundry.
It includes active pipelines, historical runs, generated results, and a
consolidated project analysis.

## Repository map

- [`Hugging Face/`](Hugging%20Face/README.md): active Transformers + PEFT/LoRA
  training, evaluation, and prediction pipeline. Its README contains setup,
  commands, configuration reference, artifacts, and reproducibility guidance.
- [`Base Model/`](Base%20Model/README.md): zero-shot evaluation of Qwen3-0.6B
  and Qwen3.5-0.8B on the Banking77 test set.
- [`Fireworks/`](Fireworks/README.md): Fireworks experiment data and preparation
  scripts.
- `MS Foundry/`: external evaluations, Banking77 agents, and Foundry-specific
  model investigations. Each maintained experiment has its own README.
- `OLD_Qwen3-0.6B/` and `OLD_Qwen3.5-0.8B/`: historical per-model training
  implementations and results.
- [`Legacy Mac Experiments/`](Legacy%20Mac%20Experiments/README.md): early local
  Hugging Face and MLX experiments.
- [`PROJECT_CONCLUSIONS.md`](PROJECT_CONCLUSIONS.md): consolidated findings and
  comparisons across the project.

New local training work should use the active [`Hugging Face/`](Hugging%20Face/README.md)
pipeline. Historical directories remain available for provenance and comparison,
but they do not define the current command-line workflow.

## Scope

The project focuses on the 77-class Banking77 intent-classification task. The
active local pipeline fine-tunes sequence-classification models with LoRA; it
is not a general text-generation or instruction-tuning framework. Other folders
evaluate alternate serving platforms or preserve earlier approaches.

Generated predictions may contain original input text and should be treated as
potentially sensitive. Model and dataset artifacts can also have licenses and
terms separate from this repository.

## License

The project code is released under the [MIT License](LICENSE). Models and
datasets retain their own licenses and terms; review their source documentation
before redistribution or production use.
