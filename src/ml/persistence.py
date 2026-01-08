"""ML Model Persistence Layer.

Responsibilities:
    - Load models from memory cache
    - Load models from disk (.joblib)
    - Save trained models
    - Enforce model-specific policies

Policy enforcement:
    - Categorizer: NEVER lazy-train (hard error if missing)
    - Forecaster: CAN lazy-train automatically
"""

from pathlib import Path
from typing import Any, Dict
from monitoring.logger import logger
import joblib
from . import categorizer, forecaster


# ---- Model Storage ----
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: Dict[str, Any] = {}


# ---- Model Policy ----
# Defines behavior for each model type
MODEL_POLICY = {
    "categorizer": {
        "lazy_train": False,  # NEVER auto-train; supervised model
        "requires": ["df"],
        "scope": "global",
    },
    "forecaster": {
        "lazy_train": True,  # CAN auto-train; self-supervised model
        "requires": ["account_id"],
        "scope": "per_account",
    },
}


class ModelNotFoundError(Exception):
    """Raised when a required model is missing and cannot be auto-trained."""
    pass


def _model_path(name: str, account_id: str | None) -> Path:
    """
    Resolve model path.

    Global models (categorizer) do not use account_id.
    Account-scoped models (forecaster) do.
    """
    if account_id is None:
        return MODEL_DIR / f"{name}.joblib"
    return MODEL_DIR / f"{name}_{account_id}.joblib"


def get_model(
    model_type: str,
    account_id: str | None = None,
) -> Any:
    """
    Load a model from memory or disk.

    Policy enforcement:
        - Categorizer: Load only. Hard error if missing.
        - Forecaster: Load or auto-train if missing.

    Args:
        model_type: "categorizer" or "forecaster"
        account_id: Required for forecaster (per-account scope)

    Returns:
        Loaded model instance

    Raises:
        ModelNotFoundError: If categorizer model is missing
        ValueError: If model_type is unknown or account_id missing for forecaster
    """

    if model_type not in MODEL_POLICY:
        raise ValueError(f"Unknown model type: {model_type}")

    policy = MODEL_POLICY[model_type]

    # ---- Validate arguments ----
    if model_type == "forecaster" and not account_id:
        raise ValueError("account_id is required for forecaster models")

    if model_type == "categorizer" and account_id:
        logger.warning("account_id ignored for categorizer model")
        account_id = None

    cache_key = (
        "categorizer:global"
        if model_type == "categorizer"
        else f"forecaster:{account_id}"
    )

    # ---- 1. In-memory cache ----
    if cache_key in _MODEL_CACHE:
        logger.debug(f"{cache_key} loaded from memory cache")
        return _MODEL_CACHE[cache_key]

    path = _model_path(model_type, account_id)

    # ---- 2. Disk ----
    if path.exists():
        model = joblib.load(path)
        _MODEL_CACHE[cache_key] = model
        logger.info(f"{cache_key} loaded from disk")
        return model

    # ---- 3. Model not found: enforce policy ----
    if not policy["lazy_train"]:
        # Categorizer: NEVER auto-train
        logger.error(
            f"{model_type} model not found. "
            "Explicit retraining required via: "
            "python -m finopsia retrain categorizer --csv <labeled_data.csv>"
        )
        raise ModelNotFoundError(
            f"{model_type} model not found at {path}. "
            "Model must be explicitly trained before running the pipeline."
        )

    # ---- 4. Auto-train (forecaster only) ----
    logger.info(f"{cache_key} not found, initiating auto-training")
    model = forecaster.train_model(account_id=account_id)

    _save_model(model, path, cache_key)
    _MODEL_CACHE[cache_key] = model

    return model


def train_and_save_model(
    model_type: str,
    account_id: str | None = None,
    **train_kwargs,
) -> Any:
    """
    Explicitly train and persist a model.

    This function is for backend-controlled retraining ONLY.
    Users should never call this directly.

    Args:
        model_type: "categorizer" or "forecaster"
        account_id: Required for forecaster
        **train_kwargs: Training arguments (e.g., df for categorizer)

    Returns:
        Trained model instance
    """

    if model_type not in MODEL_POLICY:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Explicit retraining requested for {model_type}")

    # ---- Train model ----
    if model_type == "categorizer":
        if "df" not in train_kwargs:
            raise ValueError(
                "Categorizer retraining requires labeled training data (df)"
            )
        model = categorizer.train_model(df=train_kwargs["df"])
        cache_key = "categorizer:global"
        path = _model_path(model_type, None)

    elif model_type == "forecaster":
        if not account_id:
            raise ValueError("account_id is required for forecaster retraining")
        model = forecaster.train_model(account_id=account_id)
        cache_key = f"forecaster:{account_id}"
        path = _model_path(model_type, account_id)

    # ---- Persist ----
    _save_model(model, path, cache_key)
    _MODEL_CACHE[cache_key] = model

    return model


def _save_model(model: Any, path: Path, cache_key: str) -> None:
    """Save model to disk with error handling."""
    try:
        joblib.dump(model, path)
        logger.success(f"{cache_key} model trained and saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save {cache_key} model: {e}")
        raise
