import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_classifier(config: dict, label_names: list[str], tokenizer, apply_lora: bool = True):
    id2label = dict(enumerate(label_names))
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=len(label_names), id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        dtype=getattr(torch, config.get("dtype", "bfloat16")), low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    lora = config.get("lora", {
        "enabled": True, "rank": config["lora_rank"], "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"], "target_modules": config["lora_target_modules"],
        "modules_to_save": ["score"],
    })
    if apply_lora and lora.get("enabled", True):
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
            bias="none", target_modules=list(lora["target_modules"]),
            modules_to_save=list(lora.get("modules_to_save", ["score"])),
        ))
    return model
