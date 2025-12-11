# LLM Fine-Tuning with LoRA

This project fine-tunes a small open-source language model on a custom dataset using LoRA / QLoRA with Hugging Face Transformers, TRL, and PEFT.

## Features

- Uses a 1B–7B base model (e.g., microsoft/Phi-3-mini-4k-instruct).
- Preprocesses a custom text dataset (tokenization + train/val/test split).  
- Fine-tunes the model with LoRA/QLoRA to reduce GPU memory usage.  
- Logs training metrics to Weights & Biases or TensorBoard.  
- Includes an evaluation script to compare base vs fine-tuned model.

## Project Structure

- `config/config.yaml` – config for data, model, LoRA, and training  
- `data/raw/` – place raw dataset here  
- `data/processed/` – tokenized datasets (created by preprocessing)  
- `src/preprocess.py` – preprocess + split  
- `src/train_lora.py` – LoRA/QLoRA training  
- `src/evaluate.py` – evaluation script  
- `reports/evaluation_report.md` – summary of results  
- `requirements.txt` – Python dependencies  

## Run

1. Update `config/config.yaml` for your dataset and model.  
2. `python src/preprocess.py --config config/config.yaml`  
3. `python src/train_lora.py --config config/config.yaml`  
4. `python src/evaluate.py --config config/config.yaml`  
