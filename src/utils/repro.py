"""Reproducibility utilities for deterministic execution."""

import os
import random
import torch
import numpy as np


def set_global_seed(seed: int = 2025, deterministic: bool = True) -> None:
    """
    Set global random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For Ampere+ GPUs: force deterministic algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # PyTorch < 1.8 compatibility
            pass
