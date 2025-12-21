# src/train_lora.py
import os
os.environ["BITSANDBYTES_DISABLE"] = "1"

import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from utils import load_config, set_seed


def load_model_and_tokenizer(cfg):
    """
    Load base model and tokenizer for CPU/Windows.
    Avoids bitsandbytes, 4-bit/8-bit issues.
    """
    model_name = cfg["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map=None,  # Remove device_map for CPU
      torch_dtype=torch.float32,
      low_cpu_mem_usage=False
    )
    model = model.to("cpu")  # Explicitly move to CPU


    return model, tokenizer


def main(cfg_path):
    # -----------------------------
    # 1. Load config & set seed
    # -----------------------------
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    # -----------------------------
    # 2. Load datasets
    # -----------------------------
    train_ds = load_from_disk(cfg["data"]["train_path"])
    val_ds = load_from_disk(cfg["data"]["val_path"])

    # -----------------------------
    # 3. Load model & tokenizer
    # -----------------------------
    model, tokenizer = load_model_and_tokenizer(cfg)

    # -----------------------------
    # 4. LoRA configuration
    # -----------------------------
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    # Apply LoRA (CPU-safe, no bnb)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------
    # 5. Training arguments
    # -----------------------------
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy="no",             # Save adapters manually
        report_to=train_cfg["report_to"],
        fp16=False,                     # Disable FP16 (CPU)
        bf16=False                      # Disable BF16 (CPU)
    )

    # -----------------------------
    # 6. Initialize Trainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )

    # -----------------------------
    # 7. Train the model
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 8. Save LoRA adapters & tokenizer
    # -----------------------------
    model.config.use_cache = False  # Avoid caching issues
    model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    print("LoRA training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
