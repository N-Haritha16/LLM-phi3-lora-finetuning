# src/run_eval.py
import argparse
import math
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(cfg_path):
    cfg = load_config(cfg_path)

    base_model_name = cfg["model"]["model_name"]

    # 1) Load test dataset
    test_ds = load_from_disk("data/processed/test")
    print("Columns:", test_ds.column_names)  # helpful for your report

    # 2) Load tokenizer and base model (CPU)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # 3) Compute loss and perplexity on test set
    # assumes test dataset has "input_ids" only; we use them as labels too
    model.eval()
    total_loss = 0.0
    count = 0

    for ex in test_ds:
        input_ids = torch.tensor(ex["input_ids"]).unsqueeze(0)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

        total_loss += outputs.loss.item()
        count += 1

    if count == 0:
        print("Empty test dataset.")
        return

    avg_loss = total_loss / count
    ppl = math.exp(avg_loss)

    print(f"Base model test loss: {avg_loss:.4f}")
    print(f"Base model test perplexity: {ppl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
