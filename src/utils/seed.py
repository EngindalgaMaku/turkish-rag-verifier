"""
src/utils/seed.py
=================
Reproducibility: set all random seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility.
    Call this at the start of every script.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass