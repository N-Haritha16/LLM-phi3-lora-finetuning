import argparse
import math
from typing import Tuple

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_config


def evaluate(model: AutoModelForCausalLM, dataset, device: str) -> float:
    """
    Compute perplexity of a causal LM on a tokenized dataset.

    Args:
        model: Causal language model to evaluate.
        dataset: Hugging Face Dataset containing `input_ids`.
        device: Device string, e.g. "cuda" or "cpu".

    Returns:
        Perplexity value rounded to 2 decimal places.
    """
    model.eval()
    losses = []

    for x in dataset:
        ids = torch.tensor(x["input_ids"]).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(ids, labels=ids)
            loss = outputs.loss
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return round(ppl, 2)


def load_models(cfg: dict, device: str) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM]:
    """
    Load the base model and the LoRA-adapted model for evaluation.

    This version:
    - Loads a single base model.
    - Wraps it with PeftModel for LoRA.
    - Mirrors typical QLoRA-style loading (adjust flags to match your training).
    """
    model_name = cfg["model"]["model_name"]
    lora_output_dir = cfg["evaluation"]["lora_output_dir"]

    # Mirror your training setup here. If you trained with 4‑bit / QLoRA,
    # keep load_in_4bit=True. If not, set it to False.
    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,  # set to False if training was full precision
        )
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_output_dir,
        )
        # Models are already on devices via device_map="auto"
    else:
        # Pure CPU path (slower, but avoids multi‑GPU logic)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=False,  # 4‑bit on CPU is usually not supported
        ).to(device)

        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_output_dir,
        ).to(device)

    return base_model, lora_model


def main(cfg_path: str) -> None:
    """
    Evaluate base Phi-3 and LoRA fine-tuned model on the test split.

    Steps:
      - load config and determine device,
      - load the preprocessed test dataset from disk,
      - load the base and LoRA models,
      - compute perplexity for both,
      - print base PPL, LoRA PPL, and improvement.
    """
    cfg = load_config(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load preprocessed test dataset (already tokenized)
    test_ds = load_from_disk(cfg["data"]["test_path"])

    # Ensure tokenizer exists (for completeness / future use)
    _ = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])

    # Load models
    base_model, lora_model = load_models(cfg, device)

    print("Evaluating BASE model...")
    base_ppl = evaluate(base_model, test_ds, device)
    print(f"Base PPL: {base_ppl}")

    print("\nEvaluating LoRA model...")
    lora_ppl = evaluate(lora_model, test_ds, device)
    print(f"LoRA PPL: {lora_ppl}")

    improvement = base_ppl - lora_ppl
    print(f"\nImprovement: {improvement:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
