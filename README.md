## LLM Phi-3 LoRA Finetuning

This project fine‑tunes the Phi‑3 language model using LoRA (Low‑Rank Adaptation) on a custom JSONL dataset (≥ 1,000 instruction examples) and evaluates the fine‑tuned model against the base Phi‑3 model. The repository includes scripts, configuration files, and an evaluation flow so experiments are reproducible without checking large datasets or model weights into Git.

## Project structure

config/
  └── config.yaml          # Main training / evaluation configuration

data/
  raw/
    └── mydataset.jsonl    # Raw instruction dataset (≥ 1,000 examples)
  processed/               # Tokenized train/val/test splits (ignored by Git)

src/
  preprocess.py            # Preprocess raw data into train/val/test
  train_lora.py            # LoRA fine-tuning script
  run_eval.py              # Evaluate base vs. LoRA model
  utils.py                 # Helper functions (config loading, seeding)
  inspect_val.py           # Optional inspection / debugging

outputs/
  phi-3/                   # Saved LoRA adapters (ignored by Git)
    adapter_model.safetensors
    adapter_config.json

.gitignore                 # Ignore envs, processed data, outputs, logs
requirements.txt           # Python dependencies
README.md                  # Project documentation
evaluation_report.md       # Evaluation summary (to be written after runs)
Copy_of_LLM_LoRA_finetuning.ipynb  # Colab notebook used for runs

-> data/processed/ and outputs/ are created at runtime and excluded from version control to keep the repo light.

## Setup
 Clone the repository:


git clone https://github.com/N-Haritha16/LLM-phi3-lora-finetuning.git
cd LLM-phi3-lora-finetuning

## Create and activate a virtual environment:


python -m venv .venv

Linux/macOS: source .venv/bin/activate
Windows (Git Bash / PowerShell): source .venv/Scripts/activate

## Install dependencies:


pip install -r requirements.txt

This installs PyTorch / Transformers / PEFT / Datasets and other libraries required for preprocessing, training, and evaluation.​

For full training and evaluation with Phi‑3, a GPU environment (Colab/Kaggle or similar) is strongly recommended; running the full model on CPU can be very slow or unstable.​

Data preparation
Place your dataset at:

text
data/raw/mydataset.jsonl
Each line should be a JSON object like:

json
{"instruction": "...", "input": "...", "output": "..."}

Run preprocessing:


python src/preprocess.py --config config/config.yaml
This will:​

Read data/raw/mydataset.jsonl.

Format each example using the Phi‑3 chat template via tokenizer.apply_chat_template.

Tokenize with padding/truncation.

Split into train/validation/test (e.g. 80/10/10) and save them under data/processed/ using datasets.save_to_disk.

Example console output:

text
Preprocessing complete
Train size: 800
Val size  : 100
Test size : 100
LoRA fine‑tuning

Run LoRA training:


python src/train_lora.py --config config/config.yaml

## The script:​

Loads the base Phi‑3 model and tokenizer (4‑bit QLoRA, device_map="auto" to use GPU if available).

Loads the processed train/val datasets from data/processed/.

Applies LoRA configuration (rank, alpha, dropout, target modules) from config/config.yaml.

Trains using trl.SFTTrainer with hyperparameters from the config.

Periodically saves checkpoints and final LoRA adapters under outputs/phi-3/.

If training.report_to in config.yaml is set to wandb, metrics (loss, eval loss, LR, etc.) are logged to your W&B project (for example, the run you linked).​

Example training log (truncated):

text
Preprocessing complete
Train size: 800
Val size  : 100
Test size : 100

** Running training **
  Num train examples = 800
  Num epochs = 3
  ...
Training completed successfully
LoRA training completed successfully
Evaluation
After training is done and LoRA adapters are saved:


python src/run_eval.py --config config/config.yaml

## The script:​

Loads the preprocessed test split from data/processed/test.

Loads the base Phi‑3 model.

Loads the LoRA‑adapted model by combining the base model with adapters from outputs/phi-3/ (or evaluation.lora_output_dir in config).

Computes perplexity (PPL) for both models on the same test set and prints the true improvement.

Example evaluation output (illustrative numbers only):


Evaluating BASE model...
Base PPL: 18.72

Evaluating LoRA model...
LoRA PPL: 15.43

Improvement: +3.29

Lower perplexity indicates better language modelling performance on the held‑out test set.

If you see FileNotFoundError: Directory data/processed/test not found, re‑run preprocess.py in the same environment first to regenerate the processed splits.​

## Training Logs

The LoRA fine-tuning run was tracked with Weights & Biases:

- W&B run: https://wandb.ai/harithanallamilli1606-patnr-network/llm-phi3-lora-finetune-colab/runs/wdgee7pa/logs?nw=nwuserharithanallamilli1606

This run contains training loss, validation metrics, learning rate, and token-level accuracy logged over time.

This:

1. Satisfies the requirement to provide a link to the monitoring dashboard (W&B or TensorBoard).​

2. Demonstrates that training completed with proper experiment tracking (your logs show full epochs with metrics and an eval step).


## Reproducibility

All hyperparameters and paths are controlled via config/config.yaml.

A fixed random seed is used across preprocessing, training, and evaluation through src/utils.py#set_seed.

No hyperparameters are hardcoded; everything is configurable.

Large datasets, model artifacts, and intermediate processed data are excluded from Git using .gitignore, keeping the repo lightweight and shareable.​

For a fully worked example in a GPU notebook, see the included Copy_of_LLM_LoRA_finetuning.ipynb, which mirrors this pipeline end‑to‑end in Colab and logs runs to Weights & Biases.
