"""Balance forecasting routes."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_current_user
from src.api.exceptions import BadRequestError, ForbiddenError, NotFoundError
from src.api.schemas import (
    ForecastGenerateRequest,
    ForecastGenerateResponse,
    ForecastPoint,
    ForecastResponse,
)
from src.core.currency import to_smallest_unit
from src.db.repositories import fetch_account_metadata, verify_account_ownership
from src.ml.forecaster import InsufficientDataError, forecast_balance
from src.ml.persistence import MODEL_DIR, ModelNotFoundError, get_model
from src.monitoring.logger import logger

router = APIRouter()


def _verify_account_ownership(account_id: str, user_id: int) -> None:
    """Security: Verify user owns this account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")


def _trained_at_for_model(model_type: str, account_id: str | None = None) -> str | None:
    """Return ISO timestamp for a persisted model if present."""
    path = MODEL_DIR / (
        f"{model_type}.joblib" if account_id is None else f"{model_type}_{account_id}.joblib"
    )
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat() + "Z"


def _build_forecast_response(account_id: str, user_id: int, days_ahead: int) -> ForecastResponse:
    """Generate a real forecast response using the persisted forecaster."""
    if days_ahead < 1 or days_ahead > 90:
        raise BadRequestError(
            "Invalid days_ahead",
            {"days_ahead": ["Must be between 1 and 90"]},
        )

    account_meta = fetch_account_metadata(account_id, user_id=user_id)
    model = get_model(
        model_type="forecaster",
        account_id=account_id,
        user_id=user_id,
    )
    forecast_df = forecast_balance(model, days_ahead=days_ahead)

    forecasts = [
        ForecastPoint(
            date=row.ds.date().isoformat(),
            balance=to_smallest_unit(float(row.yhat), account_meta["currency"]),
            lower_bound=to_smallest_unit(float(row.yhat_lower), account_meta["currency"]),
            upper_bound=to_smallest_unit(float(row.yhat_upper), account_meta["currency"]),
            confidence_interval=0.95,
        )
        for row in forecast_df.itertuples(index=False)
    ]

    return ForecastResponse(
        account_id=account_id,
        forecast_date=datetime.utcnow().isoformat() + "Z",
        days_ahead=days_ahead,
        forecasts=forecasts,
        model_status="trained",
        trained_at=_trained_at_for_model("forecaster", account_id),
    )


@router.get("/{account_id}/forecast", response_model=ForecastResponse)
async def get_forecast(
    account_id: str,
    days_ahead: int = 7,
    current_user: dict = Depends(get_current_user),
):
    """
    Get balance forecast for account.
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        return _build_forecast_response(account_id, current_user["user_id"], days_ahead)
    except (ForbiddenError, BadRequestError):
        raise
    except ValueError:
        raise NotFoundError("Account", account_id)
    except (InsufficientDataError, ModelNotFoundError) as exc:
        raise BadRequestError(str(exc))
    except Exception:
        logger.exception("Forecast retrieval failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{account_id}/forecast/generate", response_model=ForecastGenerateResponse)
async def generate_forecast(
    account_id: str,
    request: ForecastGenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Generate and persist a forecast for account.
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        forecast = _build_forecast_response(account_id, current_user["user_id"], request.days_ahead)
        return ForecastGenerateResponse(
            message="Forecast generated successfully",
            days_ahead=request.days_ahead,
            forecast_count=len(forecast.forecasts),
        )
    except (ForbiddenError, BadRequestError):
        raise
    except ValueError:
        raise NotFoundError("Account", account_id)
    except (InsufficientDataError, ModelNotFoundError) as exc:
        raise BadRequestError(str(exc))
    except Exception:
        logger.exception("Forecast generation failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
