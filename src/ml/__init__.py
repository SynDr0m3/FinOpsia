"""
FinOpsia ML Module.

Contains:
    - categorizer: Global transaction categorizer (CatBoost)
    - forecaster: Per-account balance forecaster (Prophet)
    - persistence: Model loading, saving, and policy enforcement
"""

from .persistence import (
    get_model,
    train_and_save_model,
    ModelNotFoundError,
    MODEL_POLICY,
)

__all__ = [
    "get_model",
    "train_and_save_model",
    "ModelNotFoundError",
    "MODEL_POLICY",
]
