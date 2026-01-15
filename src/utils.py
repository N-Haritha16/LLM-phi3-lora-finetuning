import random
import yaml
import numpy as np
import torch

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments across libraries.

    Args:
        seed: Integer seed value to use for RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
