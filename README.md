## LLM Phi-3 LoRA Finetuning

This project fine-tunes the Phi-3 language model using LoRA (Low-Rank Adaptation) on a custom JSONL dataset with at least 1,000 instruction examples and evaluates the fine-tuned model against the base Phi-3 model. The repository includes scripts, configuration files, and evaluation results, allowing reproducible experiments without storing large datasets or bulky model outputs. [file:1]

## Project Structure

config/  
  └── config.yaml          # Main training/evaluation configuration  
data/  
  raw/  
    └── mydataset.jsonl    # Raw training dataset (≥ 1,000 examples)  
  processed/               # Preprocessed dataset (ignored by Git)  
src/  
  preprocess.py            # Preprocess raw data into train/val/test  
  train_lora.py            # LoRA fine-tuning script  
  run_eval.py              # Evaluate base and LoRA models  
  utils.py                 # Helper functions (config, seeding)  
  inspect_val.py           # Optional inspection helper  
.gitignore                 # Ignore rules (envs, processed data, outputs, logs)  
evaluation_report.md       # Evaluation summary  
requirements.txt           # Python dependencies  
README.md                  # Project documentation  [file:1]

## Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

## Create and activate a virtual environment:

python -m venv .venv

Linux/macOS:


source .venv/bin/activate
Windows (Git Bash / PowerShell):


.venv\Scripts\activate
Install dependencies:


pip install -r requirements.txt
Make sure you have access to the Phi-3 model on Hugging Face and a GPU environment for training (QLoRA-style 4-bit fine-tuning is enabled by default). 

Data Preparation
Place your dataset at:

text
data/raw/mydataset.jsonl
Each line should be a JSON object with fields like:

json
{"instruction": "...", "input": "...", "output": "..."}
Run preprocessing:

python src/preprocess.py --config config/config.yaml
This will:

Read data/raw/mydataset.jsonl.

Format examples using the Phi-3 chat template via tokenizer.apply_chat_template.

Tokenize with padding/truncation and split into train/validation/test (80/10/10) under data/processed/. [file:1]

Example preprocessing output (sizes will depend on your dataset):

text

Preprocessing complete
Train size: 800
Val size  : 100
Test size : 100
LoRA Fine-Tuning
Run LoRA training:


python src/train_lora.py --config config/config.yaml
The script will:

Load the base Phi-3 model and tokenizer (4-bit, device_map="auto" for GPU usage).

Load the processed train/val datasets from data/processed/.

Apply LoRA (rank, alpha, dropout, target modules) from config/config.yaml.

Train using trl.SFTTrainer with hyperparameters defined in config/config.yaml.

Periodically save checkpoints and the final LoRA adapters in outputs/. [file:1]

When training.report_to in config.yaml is set to wandb or tensorboard, training metrics (loss, evaluation loss, learning rate, etc.) are logged to the chosen monitoring tool. [file:1]

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
Run the evaluation script:

bash
python src/run_eval.py --config config/config.yaml
The script:

Loads the preprocessed test split from data/processed/test.

Loads the base Phi-3 model and the LoRA-adapted model from outputs/phi-3 (or evaluation.lora_output_dir).

Computes perplexity (PPL) for both models and prints the true improvement without any manual adjustment. [file:1]

Example evaluation output (numbers are illustrative; your values will come from your actual run):

text

Evaluating BASE model...
Base PPL: 18.72

Evaluating LoRA model...
LoRA PPL: 15.43

Improvement: +3.29
Lower perplexity indicates better language modeling performance on the held-out test set. [file:1]

Reproducibility
All hyperparameters and paths are controlled via config/config.yaml.

A fixed random seed is used across preprocessing, training, and evaluation (src/utils.py#set_seed).

No hardcoded training or evaluation parameters; all are configurable.

Large datasets, model artifacts, and intermediate processed data are excluded from version control via .gitignore.

text

undefined

