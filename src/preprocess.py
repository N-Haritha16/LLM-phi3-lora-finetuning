# src/preprocess.py
import argparse
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer
from utils import load_config, set_seed


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_text(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
    else:
        return f"Instruction: {instruction}\nOutput: {output_text}"


def main(cfg_path):
    # --------------------------------------------------
    # 1. Load config & set seed (REPRODUCIBILITY FIX)
    # --------------------------------------------------
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # --------------------------------------------------
    # 2. Load raw dataset
    # --------------------------------------------------
    raw_records = list(load_jsonl(data_cfg["dataset_path"]))

    processed_records = [{"text": build_text(r)} for r in raw_records]
    dataset = Dataset.from_list(processed_records)

    # --------------------------------------------------
    # 3. Load tokenizer
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # 4. Tokenization
    # --------------------------------------------------
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=data_cfg["max_length"],
            truncation=True,
            padding="max_length",
        )

    tokenized_ds = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

      # --------------------------------------------------
    # 5. Train / Val / Test Split (SAFE FOR SMALL DATA)
    # --------------------------------------------------
    n = len(tokenized_ds)

    if n < 3:
        # Fallback for very small datasets (demo / testing)
        train_ds = tokenized_ds
        val_ds = tokenized_ds
        test_ds = tokenized_ds
    else:
        split_train = data_cfg["train_split"]
        split_val = data_cfg["val_split"]

        split = tokenized_ds.train_test_split(
            test_size=1 - split_train,
            seed=cfg["seed"],
        )

        remaining = split["test"]
        val_test = remaining.train_test_split(
            test_size=1 - (split_val / (1 - split_train)),
            seed=cfg["seed"],
        )

        train_ds = split["train"]
        val_ds = val_test["train"]
        test_ds = val_test["test"]


    # --------------------------------------------------
    # 6. Save processed datasets (CONFIG-DRIVEN)
    # --------------------------------------------------
    os.makedirs(data_cfg["processed_dir"], exist_ok=True)

    train_ds.save_to_disk(os.path.join(data_cfg["processed_dir"], "train"))
    val_ds.save_to_disk(os.path.join(data_cfg["processed_dir"], "val"))
    test_ds.save_to_disk(os.path.join(data_cfg["processed_dir"], "test"))

    print("Preprocessing complete")
    print(f"Train size: {len(train_ds)}")
    print(f"Val size  : {len(val_ds)}")
    print(f"Test size : {len(test_ds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
