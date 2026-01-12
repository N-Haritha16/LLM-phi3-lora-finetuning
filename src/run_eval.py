import argparse
import math
from typing import Tuple

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_config


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

    Args:
        cfg: Parsed configuration dictionary.
        device: Device string, e.g. "cuda" or "cpu".

    Returns:
        A tuple (base_model, lora_model).
    """
    model_name = cfg["model"]["model_name"]
    lora_output_dir = cfg["evaluation"]["lora_output_dir"]

    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    lora_base = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, lora_output_dir).to(device)

    return base_model, lora_model


def main(cfg_path: str) -> None:
    """
    Evaluate base Phi-3 and LoRA fine-tuned model on the test split.

    This script:
      - loads config and determines device,
      - loads the preprocessed test dataset from disk,
      - loads the base and LoRA models,
      - computes perplexity for both,
      - prints base PPL, LoRA PPL, and improvement.
    """
    cfg = load_config(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load preprocessed test dataset
    test_ds = load_from_disk(cfg["data"]["test_path"])

    # Ensure tokenizer exists (even if not used directly, for completeness)
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
