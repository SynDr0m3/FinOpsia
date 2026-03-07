import sqlite3
import pandas as pd
from src.monitoring.logger import logger
from pathlib import Path
from src.db.repositories import fetch_user_account_ids

DB_PATH = Path("data/finopsia.db")
PROCESSED_CSV = Path("data/processed/transactions_processed.csv")


def init_db():
    """Initialize the transactions table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                amount INTEGER NOT NULL,      -- FIXED
                direction TEXT CHECK(direction IN ('inflow', 'outflow')),
                currency TEXT,
                transaction_date DATE NOT NULL,
                posted_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
            """
        )

        conn.commit()
    logger.info("Database initialized")


def save_processed_csv(df: pd.DataFrame):
    """Save validated transactions to a processed CSV backup."""
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)

    # If file exists, append new rows while avoiding duplicates
    if PROCESSED_CSV.exists():
        existing = pd.read_csv(PROCESSED_CSV)
        df = pd.concat([existing, df]).drop_duplicates(subset=["transaction_id"])
    
    df.to_csv(PROCESSED_CSV, index=False)
    logger.info(f"Processed transactions saved to {PROCESSED_CSV}")


def load_transactions(df: pd.DataFrame, user_id: str) -> None:
    """
    Persist validated transactions to the database and save CSV backup.

    Args:
        df (pd.DataFrame): Validated transaction data
    """
    logger.info("Starting transaction load", extra={"account_id": None, "user_id": user_id})

    init_db()

    allowed_account_ids = set(fetch_user_account_ids(user_id))
    if not allowed_account_ids:
        raise ValueError(f"User {user_id} does not own any accounts")

    incoming_account_ids = set(df["account_id"].astype(str))
    disallowed_account_ids = sorted(incoming_account_ids - allowed_account_ids)
    if disallowed_account_ids:
        logger.error(
            f"User {user_id} attempted to load transactions for unauthorized accounts: {disallowed_account_ids}",
            extra={"account_id": None, "user_id": user_id},
        )
        raise PermissionError(
            f"User {user_id} cannot load transactions for accounts: {', '.join(disallowed_account_ids)}"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            df.to_sql(
                "transactions",
                conn,
                if_exists="append",
                index=False
            )
            logger.info(f"Loaded {len(df)} transactions to database", extra={"account_id": None, "user_id": user_id})

        except sqlite3.IntegrityError as e:
            logger.warning("Duplicate transaction detected, skipped insertion", extra={"account_id": None, "user_id": user_id})
            logger.debug(str(e), extra={"account_id": None, "user_id": user_id})

        except Exception as e:
            logger.error("Failed to load transactions", extra={"account_id": None, "user_id": user_id})
            raise

    # Save to CSV backup
    save_processed_csv(df)

