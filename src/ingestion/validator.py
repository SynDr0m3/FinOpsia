from typing import List
import pandas as pd
from loguru import logger

"""
Responsible for 
- Schema validation
- Type normalization
- Minimal feature derivation (ML-safe)
- Ensuring downstream pipeline contracts
"""

# ---- Expected Schema ----
REQUIRED_COLUMNS: List[str] = [
    "transaction_id",
    "account_id",
    "description",
    "category",
    "amount",
    "direction",
    "transaction_date",
    "posted_at",
]

ALLOWED_DIRECTIONS = {"inflow", "outflow"}


class ValidationError(Exception):
    """Raised when input data fails validation."""
    pass


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    logger.info("Validating schema...")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValidationError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data types and parse dates."""
    df = df.copy()
    logger.info("Normalizing types...")

    # Parse datetime fields
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")

    # Ensure numeric amount
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Normalize direction
    df["direction"] = df["direction"].str.lower().str.strip()

    logger.info("Type normalization complete.")
    return df


def validate_values(df: pd.DataFrame) -> None:
    """Validate value constraints."""
    logger.info("Validating values...")

    if not df["direction"].isin(ALLOWED_DIRECTIONS).all():
        invalid = df.loc[~df["direction"].isin(ALLOWED_DIRECTIONS), "direction"].unique()
        logger.error(f"Invalid direction values: {invalid}")
        raise ValidationError(f"Invalid direction values: {invalid}")

    if df["amount"].isna().any():
        logger.error("Null or non-numeric values found in 'amount'")
        raise ValidationError("Null or non-numeric values found in 'amount'")

    if df["transaction_date"].isna().any():
        logger.error("Invalid or missing 'transaction_date' values")
        raise ValidationError("Invalid or missing 'transaction_date' values")

    logger.info("Value validation passed.")


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive minimal features required by ML pipelines.
    Raw data remains untouched on disk.
    """
    df = df.copy()
    logger.info("Deriving features...")

    # Signed amount for forecasting & aggregation
    df["signed_amount"] = df.apply(
        lambda row: row["amount"] if row["direction"] == "inflow" else -row["amount"],
        axis=1,
    )

    logger.info("Feature derivation complete.")
    return df


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main validation entrypoint.

    Args:
        df (pd.DataFrame): Raw transaction data

    Returns:
        pd.DataFrame: Cleaned DataFrame with minimal derived features
    """
    logger.info("Starting transaction validation pipeline...")
    validate_schema(df)
    df = normalize_types(df)
    validate_values(df)
    df = derive_features(df)
    logger.info(f"Validation complete: {len(df)} transactions ready for downstream pipelines")
    return df