from pathlib import Path
from typing import Any, Dict
from loguru import logger
import joblib

from finopsia.ml import categorizer, forecaster

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

_MODEL_CACHE: Dict[str, Any] = {}


def _model_path(name: str) -> Path:
  return MODEL_DIR / f"{name}.joblib"


def get_model(model_type: str, training_data=None) -> Any:
    """
    Load model from cache or disk.
    If missing, trains, saves, and caches it.
    """
    # One: Memory cache
    if model_type in _MODEL_CACHE:
        return _MODEL_CACHE[model_type]

    path = _model_path(model_type)

    # Two: Disk
    if path.exists():
        model = joblib.load(path)
        _MODEL_CACHE[model_type] = model
        logger.info(f"{model_type} model loaded from disk")
        return model

    # Three: Train if missing
    if training_data is None:
        logger.error(f"No training data available for {model_type}")
        raise RuntimeError(f"Model '{model_type}' not available and cannot be trained")

    if model_type == "categorizer":
        model = categorizer.train_model(training_data)

    elif model_type == "forecaster":
        model = forecaster.train_model(training_data)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    joblib.dump(model, path)
    _MODEL_CACHE[model_type] = model

    logger.success(f"{model_type} model trained and saved")
    return model

