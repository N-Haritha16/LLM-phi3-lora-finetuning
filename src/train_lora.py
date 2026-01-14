import argparse
from typing import Tuple

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

from utils import load_config, set_seed


def load_model_and_tokenizer(cfg: dict) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """

    Load the base Phi-3 model and tokenizer (CPU or GPU if available).

    Args:
        cfg: Parsed configuration dictionary.

    Returns:
        A tuple of (model, tokenizer).

    Load the base Phi-3 model and tokenizer with optional 4-bit QLoRA support.

    If a GPU is available, the model is quantized to 4-bit using bitsandbytes
    and placed on GPU via device_map="auto".

    """
    model_name = cfg["model"]["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Decide device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # On GPU: allow 4-bit + device_map="auto"
    # On CPU: disable 4-bit and keep model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=cfg["model"].get("load_in_4bit", False) if device == "cuda" else False,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    # Decide compute dtype for GPU runs
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model_kwargs = {
        "device_map": "auto",   # use GPU(s) if available
    }

    # QLoRA path via BitsAndBytesConfig
    if cfg["model"].get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        # Non-quantized path: half precision on GPU
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model, tokenizer


def apply_lora(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    """
    Wrap the base model with LoRA adapters based on config.


    Args:
        model: Base causal language model.
        cfg: Parsed configuration dictionary.

    Returns:
        Model wrapped with LoRA using PEFT.

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

    Run LoRA fine-tuning for Phi-3 using the preprocessed dataset.

    Run LoRA fine-tuning using the preprocessed dataset.


    This script:
      - loads config and sets the seed,
      - loads preprocessed train/val datasets from disk,

      - loads the base model and tokenizer (GPU 4-bit if available),
      - applies LoRA configuration,
      - runs supervised fine-tuning with SFTTrainer,

      - loads the base model and tokenizer (optionally 4-bit QLoRA),
      - applies LoRA configuration,
      - runs supervised fine-tuning with Hugging Face Trainer,

      - saves the LoRA-adapted model and tokenizer to output_dir.
    """
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    # Load preprocessed datasets
    train_ds = load_from_disk(cfg["data"]["train_path"])
    val_ds = load_from_disk(cfg["data"]["val_path"])

    # Load model + tokenizer and apply LoRA
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)


    # Training arguments from config
        # Training arguments from config
        # Training arguments from config
    training_cfg = cfg["training"]
    eval_strategy = training_cfg.get("evaluation_strategy", "epoch")


    training_cfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],

        learning_rate=float(training_cfg["learning_rate"]),  # <-- cast to float
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=eval_strategy,
        save_strategy=training_cfg["save_strategy"],
        save_total_limit=training_cfg["save_total_limit"],
        report_to=training_cfg["report_to"],
        fp16=torch.cuda.is_available(),
    )

    trainer = SFTTrainer(

        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],   # "epoch"
        save_total_limit=training_cfg["save_total_limit"],
        report_to=training_cfg["report_to"],           # "wandb" or "tensorboard"
        fp16=training_cfg.get("fp16", False),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(

        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,

        processing_class=tokenizer,

        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    trainer.train()


    # Save LoRA adapters and tokenizer

    model.save_pretrained(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])

    print("LoRA training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
