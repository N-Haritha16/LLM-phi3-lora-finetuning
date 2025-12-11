# src/train_lora.py
import argparse
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg):
    model_name = cfg["model"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model fully on CPU (no device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def main(cfg_path):
    cfg = load_config(cfg_path)

    train_ds = load_from_disk("data/processed/train")
    val_ds = load_from_disk("data/processed/val")

    model, tokenizer = load_model_and_tokenizer(cfg)

    # 1) LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # for attention layers; adjust if needed for your Phi-3 variant
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 2) Wrap base model with LoRA
    model = get_peft_model(model, lora_config)

    # 3) Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        logging_steps=cfg["train"]["logging_steps"],
        eval_steps=cfg["train"]["eval_steps"],
        save_strategy="no",
        save_total_limit=1,
        report_to="none",
    )

    # 4) Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # 5) Train (on CPU, so it will be slow but consistent)
    trainer.train()

    # 6) Save LoRA adapters ONLY
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
