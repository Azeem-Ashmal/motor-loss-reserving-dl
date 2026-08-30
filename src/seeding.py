"""
Determinism Utilities.
Fixes every source of randomness used by the pipeline: Python's random module,
NumPy's global RNG, and PyTorch (CPU RNG, cuDNN, and algorithm selection).
Per memo.md, a bootstrap or multi-seed loop should take an explicit
numpy.random.default_rng(seed) instance rather than relying on this global state,
so two draws can never silently consume each other's randomness.
"""

import os
import random

import numpy as np
import torch


def set_deterministic_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
