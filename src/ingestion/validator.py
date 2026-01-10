from typing import List
import pandas as pd
from monitoring.logger import logger
from core.currency import to_smallest_unit


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
    logger.info("Validating schema...", extra={"account_id": None, "user_id": None})
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}", extra={"account_id": None, "user_id": None})
        raise ValidationError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.", extra={"account_id": None, "user_id": None})


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info("Normalizing types...", extra={"account_id": None, "user_id": None})

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Default currency if missing
    if "currency" not in df.columns:
        df["currency"] = "NGN"

    # Normalizing Direction
    df["direction"] = df["direction"].str.lower().str.strip()

    logger.info("Type normalization complete.", extra={"account_id": None, "user_id": None})
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
    df = df.copy()
    logger.info("Deriving features...")

    # Convert amount to smallest unit
    df["amount"] = [
        to_smallest_unit(a, c)
        for a, c in zip(df["amount"], df["currency"])
    ]

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