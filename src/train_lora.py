import argparse
from typing import Tuple, Dict, Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

from utils import load_config, set_seed


def load_model_and_tokenizer(cfg: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer, with optional 4-bit QLoRA on GPU.

    - If CUDA is available and cfg["model"].get("load_in_4bit", False) is True:
      use bitsandbytes 4-bit quantization.
    - Otherwise load in regular (half) precision on GPU, or full precision on CPU.
    """
    model_name = cfg["model"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Decide compute dtype for GPU
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    load_in_4bit = bool(cfg["model"].get("load_in_4bit", False) and device == "cuda")

    if load_in_4bit:
        # QLoRA path: 4-bit quantization with bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # Non-quantized path
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to("cpu")

    return model, tokenizer


def apply_lora(model: AutoModelForCausalLM, cfg: Dict[str, Any]) -> AutoModelForCausalLM:
    """
    Wrap the base model with LoRA adapters based on config.
    """
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


def main(cfg_path: str) -> None:
    """
    Run LoRA/QLoRA fine-tuning on preprocessed datasets.
    """
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    # Load preprocessed datasets (created by preprocess.py)
    train_ds = load_from_disk(cfg["data"]["train_path"])
    val_ds = load_from_disk(cfg["data"]["val_path"])

    # Load model + tokenizer and apply LoRA
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    training_cfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        learning_rate=float(training_cfg["learning_rate"]),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg["save_strategy"],      # e.g. "epoch"
        save_total_limit=training_cfg["save_total_limit"],
        report_to=training_cfg.get("report_to", "wandb"), # "wandb", "tensorboard", or "none"
        fp16=torch.cuda.is_available(),                   # mixed precision on GPU
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapters and tokenizer
    model.save_pretrained(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])
    print("LoRA/QLoRA training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
