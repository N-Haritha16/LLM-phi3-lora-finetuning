# src/utils.py
import yaml
import torch
import random
import numpy as np

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
