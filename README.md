## LLM Phi-3 LoRA Finetuning

This project fine-tunes the Phi-3 language model using LoRA (Low-Rank Adaptation) on a custom JSONL dataset and evaluates the fine-tuned model against the base Phi-3 model. The repository includes scripts, configuration files, and evaluation results, allowing reproducible experiments without storing large datasets or model outputs.

## Project Structure
config/
  └── config.yaml          # Main training/evaluation configuration
data/
  raw/
    └── mydataset.jsonl    # Raw training dataset
  processed/               # Preprocessed dataset (ignored by Git)
src/
  preprocess.py            # Preprocess raw data into train/val/test
  train_lora.py            # LoRA fine-tuning script
  run_eval.py              # Evaluate base and LoRA models
  utils.py                 # Helper functions (logging, seeding, etc.)
.gitignore                 # Ignore rules (envs, processed data, outputs, logs)
evaluation_report.md       # Evaluation summary
requirements.txt           # Python dependencies
README.md                  # Project documentation

## Setup

# Clone the repository:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>


# Create and activate a virtual environment:

python -m venv .venv


Linux/macOS:

source .venv/bin/activate


Windows (Git Bash / PowerShell):

.venv\Scripts\activate


# Install dependencies:

pip install -r requirements.txt


Make sure you have access to the Phi-3 model (e.g., Hugging Face) and a GPU environment for training.

# Data Preparation

Place your dataset at:

data/raw/mydataset.jsonl


Each line should be a JSON object with fields like instruction, input, output.

# Run preprocessing:

python src/preprocess.py

This will:

Read data/raw/mydataset.jsonl

Clean and transform data for Phi-3 LoRA finetuning

Split into train/validation/test sets under data/processed/

## LoRA Fine-Tuning

Run LoRA training:

python src/train_lora.py --config config/config.yaml


Steps:

Load base Phi-3 model and tokenizer

Load processed dataset

Apply LoRA (rank, alpha, dropout, target modules)

Train according to config.yaml hyperparameters

Save LoRA adapters/fine-tuned model in outputs/

Adjust hyperparameters, dataset paths, and model name in config.yaml.

## Evaluation

Run the evaluation script:

python src/run_eval.py --config config/config.yaml


The script:

Loads the test split

Evaluates base Phi-3 and LoRA fine-tuned model

Computes loss, perplexity (PPL) and writes evaluation_report.md

# Example Output:

Preprocessing complete
Train size: 2
Val size  : 2
Test size : 2

Trainable params: ~1.2M
Training completed successfully

Evaluating BASE model...
Evaluating LoRA model...

Evaluation completed successfully
Base PPL: 14.11
LoRA PPL: 9.77
Improvement: +4.34

## Evaluation Results

Evaluating BASE model...
Base PPL: 14.11

Evaluating LoRA model...
LoRA PPL: 9.77

Improvement: +4.34

Lower perplexity indicates better language modeling performance.



## Reproducibility

- All hyperparameters are controlled via `config/config.yaml`
- Fixed random seed across preprocessing, training, and evaluation
- No hardcoded training or evaluation parameters
- Large datasets and model artifacts excluded from version control
