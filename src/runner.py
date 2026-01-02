"""
End-to-end FinOpsia pipeline runner.

Flow:
    CSV → validate → categorize → persist (DB + CSV)
    Optional: forecast balances
"""

from pathlib import Path
from loguru import logger
import pandas as pd

# Ingestion
from ingestion.reader import read_transactions
from ingestion.validator import validate_transactions
from ingestion.loader import load_transactions

# ML
from ml.persistence import get_model, ModelNotFoundError
from ml import categorizer
from ml.forecaster import forecast_balance

# Defaults
DEFAULT_CSV_PATH = Path("data/raw/transactions.csv")
DEFAULT_FORECAST_DAYS = 7

    
def run_pipeline(
    csv_path: Path,
    *,
    forecast: bool = False,
    account_id: str | None = None,
    forecast_days: int = DEFAULT_FORECAST_DAYS,
    dry_run: bool = False,
) -> None:
    """
    Run the full FinOpsia pipeline.

    This is inference-only mode:
        - Categorizer model MUST exist (no training)
        - Forecaster will auto-train if missing

    Steps:
        1. Read raw transactions from CSV
        2. Validate transactions
        3. Categorize transactions (using existing model)
        4. Persist categorized data (DB + processed CSV)
        5. Optionally forecast balances

    Args:
        csv_path: Path to raw transactions CSV
        forecast: Whether to run balance forecasting
        account_id: Required if forecast=True
        forecast_days: Number of days to forecast
        dry_run: If True, skip persistence

    Raises:
        ModelNotFoundError: If categorizer model is missing
    """

    logger.info("Starting FinOpsia pipeline")

    # -----------------------------
    # 1. Read
    # -----------------------------
    df_raw = read_transactions(csv_path)
    logger.info(f"Read {len(df_raw)} transactions from {csv_path}")

    # -----------------------------
    # 2. Validate
    # -----------------------------
    df_raw = validate_transactions(df_raw)
    logger.info(f"{len(df_raw)} transactions after validation")

    # -----------------------------
    # 3. Categorize (MANDATORY)
    # -----------------------------
    # NOTE: Categorizer model MUST exist.
    # No lazy-training allowed — model writes to DB.
    # If missing, user must run: python -m finopsia retrain categorizer --csv <data>

    logger.info("Loading categorizer model (inference-only)")

    categorizer_model = get_model(model_type="categorizer")

    logger.info("Categorizing transactions")
    df_categorized = categorizer.predict(
        df=df_raw,
        model=categorizer_model,
    )

    logger.success("Transaction categorization completed")

    # -----------------------------
    # 4. Persist (DB + CSV)
    # -----------------------------
    
    if dry_run:
      logger.warning("Dry-run enabled: skipping DB and CSV persistence")
    else:
      load_transactions(df_categorized)
      logger.success("Transactions saved to database and processed CSV")


    # -----------------------------
    # 5. Optional Forecasting
    # -----------------------------
    # NOTE: Forecaster is per-account and self-supervised.
    # Auto-training is allowed if model is missing.

    if forecast:
        if not account_id:
            raise ValueError(
                "account_id is required when forecast=True"
            )

        logger.info(
            f"Loading forecaster model for account {account_id}"
        )

        # Forecaster will auto-train if missing (per MODEL_POLICY)
        forecaster_model = get_model(
            model_type="forecaster",
            account_id=account_id,
        )

        logger.info(
            f"Forecasting balances for next {forecast_days} days"
        )

        forecast_df = forecast_balance(
            model=forecaster_model,
            days_ahead=forecast_days,
        )

        logger.success("Forecasting completed")
        logger.debug(f"\n{forecast_df}")

    logger.success("FinOpsia pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline(csv_path=DEFAULT_CSV_PATH)
