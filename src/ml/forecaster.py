from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.prophet_runtime import ensure_cmdstan_runtime
from src.monitoring.logger import logger

try:
    ensure_cmdstan_runtime()
    from prophet import Prophet as ProphetModel
except Exception:  # pragma: no cover - optional runtime path
    ProphetModel = None

from src.core.currency import from_smallest_unit
from src.db.repositories import fetch_transactions, fetch_account_metadata

MIN_DAYS_REQUIRED = 90
DEFAULT_LOOKBACK_YEARS = 2
MAX_LOOKBACK_YEARS = 5
DEFAULT_TRAINING_BACKEND = os.environ.get("FINOPSIA_FORECAST_BACKEND", "native").strip().lower()


class InsufficientDataError(Exception):
    """Raised when there is not enough historical data to train a model."""
    pass


@dataclass
class NativeBalanceForecaster:
    """Lightweight daily balance forecaster with the Prophet prediction interface."""

    history: pd.DataFrame
    intercept: float
    slope: float
    weekday_offsets: dict[int, float]
    sigma: float

    def make_future_dataframe(self, periods: int) -> pd.DataFrame:
        history_dates = pd.to_datetime(self.history["ds"])
        future_dates = pd.date_range(
            start=history_dates.iloc[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq="D",
        )
        return pd.DataFrame({"ds": list(history_dates) + list(future_dates)})

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        ds = pd.to_datetime(future["ds"])
        origin = pd.Timestamp(self.history["ds"].iloc[0])
        day_index = (ds - origin).dt.days.astype(float)
        weekday_component = ds.dt.dayofweek.map(lambda day: self.weekday_offsets.get(int(day), 0.0)).astype(float)
        yhat = self.intercept + (self.slope * day_index) + weekday_component
        interval = max(float(self.sigma), 1e-6) * 1.96

        return pd.DataFrame(
            {
                "ds": ds,
                "yhat": yhat,
                "yhat_lower": yhat - interval,
                "yhat_upper": yhat + interval,
            }
        )


def _signed_amount(row: pd.Series) -> int:
    """
    Convert inflow/outflow into signed amount (smallest unit).
    """
    return row["amount"] if row["direction"] == "inflow" else -row["amount"]


def build_daily_balance_series(account_id: str, user_id: str) -> pd.DataFrame:
    """
    Build daily balance time series for forecasting.

    Returns DataFrame with:
    - ds: date
    - y: balance (major currency unit)
    """

    logger.info(f"Building daily balance series for account {account_id}", extra={"account_id": account_id, "user_id": user_id})

    meta = fetch_account_metadata(account_id, user_id=user_id)
    currency = meta["currency"]
    starting_balance = meta["starting_balance"]  # smallest unit

    lookback_years = DEFAULT_LOOKBACK_YEARS
    daily_cashflow = pd.Series(dtype="int64")

    while lookback_years <= MAX_LOOKBACK_YEARS:
        txns = fetch_transactions(
            account_id=account_id,
            years_back=lookback_years,
            user_id=user_id,
        )

        if txns.empty:
            logger.warning(
                f"No transactions found for last {lookback_years} years", extra={"account_id": account_id, "user_id": user_id}
            )
            lookback_years += 1
            continue

        txns["signed_amount"] = txns.apply(_signed_amount, axis=1)

        daily_cashflow = (
            txns
            .groupby("txn_date")["signed_amount"]
            .sum()
            .sort_index()
        )

        if len(daily_cashflow) >= MIN_DAYS_REQUIRED:
            break

        lookback_years += 1

    if daily_cashflow.empty or len(daily_cashflow) < MIN_DAYS_REQUIRED:
        logger.error(
            f"Insufficient data to train forecaster for account {account_id}", extra={"account_id": account_id, "user_id": user_id}
        )
        raise InsufficientDataError(
            f"Not enough data to train model for account {account_id}"
        )

    daily_balance = daily_cashflow.cumsum() + starting_balance

    df = pd.DataFrame({
        "ds": daily_balance.index,
        "y": daily_balance.apply(
            lambda x: from_smallest_unit(x, currency)
        ),
    })

    logger.success(
        f"Built balance series with {len(df)} days of data", extra={"account_id": account_id, "user_id": user_id}
    )

    return df


def train_model(account_id: str, user_id: str) -> Prophet:
    """
    Train a Prophet balance forecasting model.
    """

    logger.info(f"Training balance forecaster for account {account_id}", extra={"account_id": account_id, "user_id": user_id})

    df = build_daily_balance_series(account_id, user_id)

    if DEFAULT_TRAINING_BACKEND == "prophet":
        if ProphetModel is None:
            raise RuntimeError("Prophet training backend requested, but Prophet runtime is unavailable")

        model = ProphetModel(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.fit(df)
    else:
        model = _train_native_model(df)

    logger.success("Balance forecaster training completed", extra={"account_id": account_id, "user_id": user_id})

    return model


def _train_native_model(df: pd.DataFrame) -> NativeBalanceForecaster:
    """Train a deterministic balance forecaster that matches Prophet's prediction interface."""
    history = df.sort_values("ds").reset_index(drop=True).copy()
    history["ds"] = pd.to_datetime(history["ds"])

    day_index = np.arange(len(history), dtype=float)
    slope, intercept = np.polyfit(day_index, history["y"].astype(float).to_numpy(), 1)

    baseline = intercept + (slope * day_index)
    residuals = history["y"].astype(float).to_numpy() - baseline
    weekday_series = history["ds"].dt.dayofweek
    weekday_offsets = {
        int(day): float(np.mean(residuals[weekday_series.to_numpy() == day]))
        for day in range(7)
        if np.any(weekday_series.to_numpy() == day)
    }
    seasonal_residual = np.array([weekday_offsets.get(int(day), 0.0) for day in weekday_series], dtype=float)
    sigma = float(np.std(residuals - seasonal_residual, ddof=1)) if len(history) > 1 else 0.0

    logger.info("Using native forecaster backend", extra={"account_id": None, "user_id": None})
    return NativeBalanceForecaster(
        history=history[["ds", "y"]].copy(),
        intercept=float(intercept),
        slope=float(slope),
        weekday_offsets=weekday_offsets,
        sigma=sigma,
    )


def forecast_balance(
    model,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Forecast future balances using a trained model.
    """

    logger.info(f"Forecasting balance for next {days_ahead} days", extra={"account_id": None, "user_id": None})

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    return forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(days_ahead)
