import argparse
import json
import os
from typing import Dict, Iterable, Any, List

from datasets import Dataset
from transformers import AutoTokenizer

from utils import load_config, set_seed


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    Stream a JSONL file line by line.

    Args:
        path: Path to the .jsonl file.

    Yields:
        Parsed JSON object for each line.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_text(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """
    Build a chat-formatted text prompt for Phi-3 using its chat template.

    Args:
        example: A single data example containing `instruction`, optional
                 `input`, and `output` fields.
        tokenizer: Tokenizer for the Phi-3 model.

    Returns:
        Formatted text string using the model's chat template.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Concatenate instruction and input for the user message
    if input_text:
        user_content = instruction + "\n" + input_text
    else:
        user_content = instruction

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def main(cfg_path: str) -> None:
    """
    Preprocess the raw JSONL dataset into train/val/test splits for Phi-3 LoRA.

    This script:
      - loads the raw JSONL file,
      - formats each example using the Phi-3 chat template,
      - tokenizes with padding and truncation,
      - creates an 80/10/10 train/val/test split,
      - saves splits under `data/processed/*`.

    Args:
        cfg_path: Path to the YAML configuration file.
    """
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    # Load raw JSONL records
    records = list(load_jsonl(cfg["data"]["dataset_path"]))
    if not records:
        raise ValueError(f"No records found in {cfg['data']['dataset_path']}")



    # OPTIONAL STRICT CHECK (uncomment if you want to enforce >=1000)
    # if len(records) < 1000:
    #     raise ValueError(
    #         f"Dataset too small: {len(records)} examples found, "
    #         "but at least 1000 are required."
    #     )


    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build text field using chat template
    dataset = Dataset.from_list(
        [{"text": build_text(r, tokenizer)} for r in records]
    )

    # Tokenize
    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg["data"]["max_length"],
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Train/val/test split: 80/10/10
    train_split = cfg["data"].get("train_split", 0.8)
    val_split = cfg["data"].get("val_split", 0.1)
    # Remaining goes to test
    test_split = 1.0 - train_split - val_split
    if test_split <= 0:
        raise ValueError(
            f"Invalid splits: train={train_split}, val={val_split}. "
            "Their sum must be < 1.0."
        )

    split = tokenized.train_test_split(
        test_size=(1.0 - train_split),
        seed=cfg["seed"],
    )
    val_test = split["test"].train_test_split(
        test_size=test_split / (val_split + test_split),
        seed=cfg["seed"],
    )

    train_ds = split["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)

    train_ds.save_to_disk(cfg["data"]["train_path"])
    val_ds.save_to_disk(cfg["data"]["val_path"])
    test_ds.save_to_disk(cfg["data"]["test_path"])

    print("Preprocessing complete")
    print("Train size:", len(train_ds))
    print("Val size  :", len(val_ds))
    print("Test size :", len(test_ds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
