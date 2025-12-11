import argparse
import json
import os
import random

from datasets import Dataset
from transformers import AutoTokenizer
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(cfg_path):
    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    raw_path = data_cfg["dataset_path"]
    records = list(load_jsonl(raw_path))

    # Build a single "text" field from instruction + input + output
    processed_records = []
    for r in records:
        instruction = r.get("instruction", "")
        input_text = r.get("input", "")
        output_text = r.get("output", "")
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output_text}"
        processed_records.append({"text": text})

    dataset = Dataset.from_list(processed_records)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=data_cfg["max_length"],
            truncation=True,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # Split into train / val / test
    split_train = data_cfg["train_split"]
    split_val = data_cfg["val_split"]
    n = len(tokenized)
    idx = list(range(n))
    random.shuffle(idx)
    train_end = int(split_train * n)
    val_end = int((split_train + split_val) * n)

    train_ds = tokenized.select(idx[:train_end])
    val_ds = tokenized.select(idx[train_end:val_end])
    test_ds = tokenized.select(idx[val_end:])

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    train_ds.save_to_disk("data/processed/train")
    val_ds.save_to_disk("data/processed/val")
    test_ds.save_to_disk("data/processed/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
