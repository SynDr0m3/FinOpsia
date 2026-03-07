"""Model management and retraining routes."""
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_current_admin, get_current_user
from src.api.exceptions import BadRequestError, ForbiddenError, NotFoundError
from src.api.schemas import (
    CategorizerDetailsResponse,
    CategorizerRetrainRequest,
    CategorizerStatus,
    ForecasterStatus,
    ModelStatusResponse,
    RetrainingJobResponse,
)
from src.db.repositories import (
    fetch_account_owner_id,
    fetch_transactions,
    fetch_user_account_ids,
    verify_account_ownership,
)
from src.ml import persistence as model_persistence
from src.ml.forecaster import InsufficientDataError
from src.ml.persistence import MODEL_DIR, train_and_save_model
from src.monitoring.logger import logger

router = APIRouter()

JOBS: dict[str, dict] = {}


def _verify_account_ownership(account_id: str, user_id: int) -> None:
    """Ensure the authenticated user owns the target account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")


def _model_file(model_type: str, account_id: str | None = None) -> Path:
    """Resolve a persisted model path."""
    return MODEL_DIR / (
        f"{model_type}.joblib" if account_id is None else f"{model_type}_{account_id}.joblib"
    )


def _trained_at(path: Path) -> str | None:
    """Return model file mtime in ISO format."""
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat() + "Z"


def _categorizer_details() -> CategorizerDetailsResponse:
    """Build categorizer metadata from the real artifact on disk."""
    path = _model_file("categorizer")
    status_value = "trained" if path.exists() else "missing"
    model = model_persistence.get_model("categorizer") if path.exists() else None
    categories = sorted(model.classes_.tolist()) if model is not None and hasattr(model, "classes_") else []

    return CategorizerDetailsResponse(
        model_type="categorizer",
        status=status_value,
        trained_at=_trained_at(path),
        version=f"joblib:{int(path.stat().st_mtime)}" if path.exists() else "untrained",
        categories=categories,
        performance={},
        training_samples=0,
    )


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(current_user: dict = Depends(get_current_user)):
    """
    Get status of real trained models visible to the current user.
    """
    try:
        categorizer = _categorizer_details()
        account_ids = fetch_user_account_ids(current_user["user_id"])

        forecasters = []
        for account_id in account_ids:
            path = _model_file("forecaster", account_id)
            txns = fetch_transactions(account_id, user_id=current_user["user_id"])
            forecasters.append(
                ForecasterStatus(
                    account_id=account_id,
                    status="trained" if path.exists() else "missing",
                    trained_at=_trained_at(path),
                    days_history=len(txns),
                    mape=None,
                    last_error=None if path.exists() else "Model not trained yet",
                )
            )

        return ModelStatusResponse(
            categorizer=CategorizerStatus(
                status=categorizer.status,
                trained_at=categorizer.trained_at,
                version=categorizer.version,
                accuracy=None,
            ),
            forecasters=forecasters,
        )
    except Exception:
        logger.exception("Model status retrieval failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/categorizer", response_model=CategorizerDetailsResponse)
async def get_categorizer_details(current_user: dict = Depends(get_current_user)):
    """
    Get real categorizer metadata from the persisted artifact.
    """
    try:
        return _categorizer_details()
    except Exception:
        logger.exception("Categorizer details retrieval failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/categorizer/retrain", response_model=RetrainingJobResponse, status_code=status.HTTP_200_OK)
async def retrain_categorizer(
    request: CategorizerRetrainRequest,
    current_user: dict = Depends(get_current_admin),
):
    """
    Retrain the categorizer immediately using supplied training samples.
    """
    job_id = f"categorizer_{datetime.utcnow().timestamp()}"
    try:
        training_df = pd.DataFrame(
            [{"description": sample.description, "category": str(sample.category)} for sample in request.training_samples]
        )
        train_and_save_model(
            model_type="categorizer",
            df=training_df,
        )
        trained_at = _trained_at(_model_file("categorizer"))
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "message": "Categorizer retraining completed",
            "trained_at": trained_at,
            "performance": {},
        }
        return RetrainingJobResponse(**JOBS[job_id])
    except Exception as exc:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "message": str(exc),
        }
        logger.exception("Categorizer retraining failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        return RetrainingJobResponse(**JOBS[job_id])


@router.get("/categorizer/retrain/{job_id}", response_model=RetrainingJobResponse)
async def get_retrain_status(
    job_id: str,
    current_user: dict = Depends(get_current_admin),
):
    """
    Return the real recorded result of the last retrain request.
    """
    if job_id not in JOBS:
        raise NotFoundError("Job", job_id)
    return RetrainingJobResponse(**JOBS[job_id])


@router.post("/{account_id}/forecast/retrain", response_model=RetrainingJobResponse)
async def retrain_forecaster(
    account_id: str,
    current_user: dict = Depends(get_current_admin),
):
    """
    Retrain a real forecaster model for an account.
    """
    job_id = f"forecaster_{account_id}_{datetime.utcnow().timestamp()}"
    try:
        owner_id = fetch_account_owner_id(account_id)
        train_and_save_model(
            model_type="forecaster",
            account_id=account_id,
            user_id=owner_id,
        )
        JOBS[job_id] = {
            "job_id": job_id,
            "account_id": account_id,
            "status": "completed",
            "message": f"Forecaster retraining completed for account {account_id}",
            "trained_at": _trained_at(_model_file("forecaster", account_id)),
        }
        return RetrainingJobResponse(**JOBS[job_id])
    except ValueError:
        raise NotFoundError("Account", account_id)
    except InsufficientDataError as exc:
        JOBS[job_id] = {
            "job_id": job_id,
            "account_id": account_id,
            "status": "failed",
            "message": str(exc),
        }
        return RetrainingJobResponse(**JOBS[job_id])
    except Exception as exc:
        JOBS[job_id] = {
            "job_id": job_id,
            "account_id": account_id,
            "status": "failed",
            "message": str(exc),
        }
        logger.exception("Forecaster retraining failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        return RetrainingJobResponse(**JOBS[job_id])


@router.get("/{account_id}/forecast/retrain/{job_id}", response_model=RetrainingJobResponse)
async def get_forecaster_retrain_status(
    account_id: str,
    job_id: str,
    current_user: dict = Depends(get_current_admin),
):
    """
    Return the recorded result for a forecaster retrain request.
    """
    if job_id not in JOBS:
        raise NotFoundError("Job", job_id)
    job = JOBS[job_id]
    if job.get("account_id") != account_id:
        raise NotFoundError("Job", job_id)
    return RetrainingJobResponse(**job)


@router.delete("/{account_id}/forecaster", status_code=status.HTTP_204_NO_CONTENT)
async def delete_forecaster(
    account_id: str,
    current_user: dict = Depends(get_current_admin),
):
    """
    Delete the real persisted forecaster artifact for an account.
    """
    path = _model_file("forecaster", account_id)
    if not path.exists():
        raise NotFoundError("Forecaster", account_id)

    try:
        path.unlink()
        model_persistence._MODEL_CACHE.pop(f"forecaster:{account_id}", None)
        return None
    except Exception:
        logger.exception("Forecaster deletion failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
