from pathlib import Path
from typing import Any, Dict
from loguru import logger
import joblib
from . import categorizer, forecaster

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE: Dict[str, Any] = {}

def _model_path(name: str, account_id: str) -> Path:
    return MODEL_DIR / f"{name}_{account_id}.joblib"

def get_model(model_type: str, account_id: str, **train_kwargs) -> Any:
    
    """
    Load a model from memory or disk. If missing, train, persist, and cache it.

    Args:
        model_type (str): Either "categorizer" or "forecaster".
        account_id (str): Account id.
        **train_kwargs: Arguments to pass to the train_model function if training is needed.
            For categorizer: requires `df` keyword argument.
            For forecaster: requires `account_id` keyword argument.

    Returns:
        Any: Trained or loaded model instance.

    Raises:
        ValueError: If an unknown model_type is requested.
    """

    cache_key = f"{model_type}:{account_id}"

    # 1. In-memory cache
    if cache_key in _MODEL_CACHE:
        logger.debug(f"{cache_key} model retrieved from memory cache")
        return _MODEL_CACHE[cache_key]

    path = _model_path(model_type, account_id)

    # 2. Disk
    if path.exists():
        model = joblib.load(path)
        _MODEL_CACHE[cache_key] = model
        logger.info(f"{cache_key} model loaded from disk")
        return model

        # 3. Lazy training
    if model_type == "categorizer":
        # Requires `df` keyword argument in train_kwargs
        model = categorizer.train_model(**train_kwargs)
    elif model_type == "forecaster":
        # Requires `account_id` keyword argument in train_kwargs
        model = forecaster.train_model(**train_kwargs)
    else:
        logger.error(f"Unknown model type requested: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        joblib.dump(model, path)
        logger.success(f"{model_type} model trained and saved")
    except Exception as e:
        logger.error(f"Failed to save {model_type} model: {e}")
        raise

    _MODEL_CACHE[cache_key] = model
    return model
