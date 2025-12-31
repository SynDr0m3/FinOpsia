from datetime import timedelta
import pandas as pd
from prophet import Prophet
from loguru import logger

from core.currency import from_smallest_unit
from db.repositories import fetch_transactions, fetch_account_metadata

MIN_DAYS_REQUIRED = 90
DEFAULT_LOOKBACK_YEARS = 2
MAX_LOOKBACK_YEARS = 5


class InsufficientDataError(Exception):
    """Raised when there is not enough historical data to train a model."""
    pass


def _signed_amount(row: pd.Series) -> int:
    """
    Convert inflow/outflow into signed amount (smallest unit).
    """
    return row["amount"] if row["direction"] == "inflow" else -row["amount"]


def build_daily_balance_series(account_id: str) -> pd.DataFrame:
    """
    Build daily balance time series for forecasting.

    Returns DataFrame with:
    - ds: date
    - y: balance (major currency unit)
    """

    logger.info(f"Building daily balance series for account {account_id}")

    meta = fetch_account_metadata(account_id)
    currency = meta["currency"]
    starting_balance = meta["starting_balance"]  # smallest unit

    lookback_years = DEFAULT_LOOKBACK_YEARS
    daily_cashflow = pd.Series(dtype="int64")

    while lookback_years <= MAX_LOOKBACK_YEARS:
        txns = fetch_transactions(
            account_id=account_id,
            years_back=lookback_years,
        )

        if txns.empty:
            logger.warning(
                f"No transactions found for last {lookback_years} years"
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
            f"Insufficient data to train forecaster for account {account_id}"
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
        f"Built balance series with {len(df)} days of data"
    )

    return df


def train_model(account_id: str) -> Prophet:
    """
    Train a Prophet balance forecasting model.
    """

    logger.info(f"Training balance forecaster for account {account_id}")

    df = build_daily_balance_series(account_id)

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )

    model.fit(df)

    logger.success("Balance forecaster training completed")

    return model


def forecast_balance(
    model: Prophet,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Forecast future balances using a trained model.
    """

    logger.info(f"Forecasting balance for next {days_ahead} days")

    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)

    return forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(days_ahead)
