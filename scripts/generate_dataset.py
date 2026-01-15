import os
import json
import random

OUTPUT_PATH = "data/raw/mydataset.jsonl"
NUM_EXAMPLES = 1001 # >= 1000


PYTHON_INSTRUCTIONS = [
    (
        "Write a Python function to reverse a string.",
        "",
        "def reverse_string(s: str) -> str:\n    return s[::-1]"
    ),
    (
        "Fix the bug in this Python function so it correctly sums a list of numbers.",
        "def sum_list(xs):\n    total = 1\n    for x in xs:\n        total += x\n    return total",
        "def sum_list(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total"
    ),
    (
        "Write a Python function that checks if a number is prime.",
        "",
        "def is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True"
    ),
]

DOCKER_INSTRUCTIONS = [
    (
        "Create a Dockerfile for a FastAPI app in main.py listening on port 8000.",
        "",
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]"
    ),
    (
        "Explain what a Docker volume is and when to use it.",
        "",
        "A Docker volume is a persistent storage mechanism managed by Docker. "
        "Use it to store data outside the container filesystem so it survives container restarts "
        "and image updates, and to share data between containers."
    ),
    (
        "Write a docker-compose service definition for a Postgres database named db.",
        "",
        "services:\n  db:\n    image: postgres:15\n    environment:\n      POSTGRES_USER: user\n      POSTGRES_PASSWORD: password\n      POSTGRES_DB: appdb\n    ports:\n      - \"5432:5432\"\n    volumes:\n      - db_data:/var/lib/postgresql/data\n\nvolumes:\n  db_data:"
    ),
]

ML_INSTRUCTIONS = [
    (
        "Explain the concept of overfitting in machine learning.",
        "",
        "Overfitting occurs when a model learns patterns specific to the training data, including noise, "
        "and fails to generalize to unseen data. It appears as low training loss but high validation loss."
    ),
    (
        "Given the following training behavior, identify a sign of overfitting and one mitigation.",
        "Train loss keeps decreasing while validation loss starts increasing after epoch 3.",
        "This is a sign of overfitting because the model continues improving on the training data while "
        "performance on validation data degrades. A mitigation is to use early stopping around epoch 3 "
        "or add regularization such as dropout or weight decay."
    ),
    (
        "What is perplexity in language modeling, and is lower or higher better?",
        "",
        "Perplexity is the exponential of the average negative log-likelihood of a language model. "
        "It measures how well the model predicts a sequence of tokens. Lower perplexity indicates a better model."
    ),
]


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_examples = []
    pools = [PYTHON_INSTRUCTIONS, DOCKER_INSTRUCTIONS, ML_INSTRUCTIONS]

    for _ in range(NUM_EXAMPLES):
        pool = random.choice(pools)
        inst, inp, out = random.choice(pool)
        all_examples.append(
            {
                "instruction": inst,
                "input": inp,
                "output": out,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
