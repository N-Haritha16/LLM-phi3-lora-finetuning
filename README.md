# LLM Phi-3 LoRA Finetuning

This project fine-tunes the Phi-3 language model using LoRA on a custom JSONL dataset and evaluates the fine-tuned model against the base model. The repository includes code, configuration, raw data, and reports so experiments can be reproduced without storing large processed datasets, logs, or model outputs. 

## Project Structure

config/ # Additional configuration files (if any)
config.yaml # Main training/evaluation configuration
data/
raw/
mydataset.jsonl # Raw training data in JSONL format
src/
preprocess.py # Script to preprocess raw data into train/val/test
train_lora.py # Script to fine-tune Phi-3 using LoRA
run_eval.py # Script to evaluate base and fine-tuned models
utils.py # Shared helper functions (logging, seeding, etc.)
.gitignore # Ignore rules (envs, processed data, logs, outputs, etc.)
README.md # Project documentation (this file)
evaluation_report.md # Final evaluation results and discussion
requirements.txt # Python dependencies

## Setup

1. Clone the repository:

  git clone https://github.com/<your-username>/<your-repo>.git    
     

2. Create and activate a virtual environment:

python -m venv .venv

Linux/macOS
source .venv/bin/activate

Windows (Git Bash / PowerShell)
.venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt
Ensure you have access to the Phi-3 model you plan to fine-tune (for example via Hugging Face) and a suitable GPU environment for training.

## Data Preparation

Place your raw dataset as a JSONL file at:

data/raw/mydataset.jsonl

Each line should contain one training example in JSON format, with the fields expected by `src/preprocess.py` (for example, instruction/input/output fields for instruction tuning). 

Run the preprocessing script:

python src/preprocess.py

This will:

- Read `data/raw/mydataset.jsonl`.  
- Clean and transform the data into the format required for Phi-3 LoRA finetuning.  
- Split into train/validation/test splits and save them under `data/processed/` (ignored by Git). 

## Training with LoRA

After preprocessing completes, start LoRA finetuning:

python src/train_lora.py --config config.yaml

The training script typically:

- Loads the base Phi-3 model and tokenizer.  
- Loads the processed dataset from `data/processed/`.  
- Applies LoRA (rank, alpha, dropout, target modules).  
- Trains according to hyperparameters in `config.yaml` (epochs, batch size, learning rate, etc.).  
- Saves the resulting LoRA adapters or fine-tuned model into an outputs directory (ignored by Git). 

Edit `config.yaml` to adjust model name/path, dataset paths, LoRA parameters, and training hyperparameters.

## Evaluation

To evaluate the base and fine-tuned models:

python src/run_eval.py --config config.yaml



The evaluation script:

- Loads the test split from `data/processed/`.  
- Evaluates both the base Phi-3 model and the fine-tuned model on the same test set.  
- Computes metrics such as loss and perplexity, and writes a summary (for example reflected in `evaluation_report.md`). 

Use `evaluation_report.md` to document:

- Test loss and perplexity for base vs. fine-tuned models.  
- Observations about performance improvements, failure cases, and future work.

## Reproducibility and Git Hygiene

The repository is configured (via `.gitignore`) to **exclude**:

- Large processed datasets (`data/processed/`).  
- Logs and experiment artifacts (`logs/`, `wandb/`).  
- Model outputs and checkpoints (`outputs/`).  
- Local environments and notebook checkpoints.

Anyone can reproduce these artifacts by:

1. Installing dependencies (`requirements.txt`).  
2. Running `src/preprocess.py`, `src/train_lora.py`, and `src/run_eval.py` with the same `config.yaml` and `data/raw/mydataset.jsonl`. 

For fully reproducible experiments, set random seeds in your code (for example in `utils.py`) and keep important hyperparameters and results recorded in `config.yaml` and `evaluation_report.md`. [web:387][web:384]
