import sqlite3
import pandas as pd
from monitoring.logger import logger
from pathlib import Path

DB_PATH = Path("data/finopsia.db")
PROCESSED_CSV = Path("data/processed/transactions_processed.csv")


def init_db():
    """Initialize the transactions table if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                amount INTEGER NOT NULL,      -- FIXED
                direction TEXT CHECK(direction IN ('inflow', 'outflow')),
                transaction_date DATE NOT NULL,
                posted_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def load_transactions(df: pd.DataFrame) -> None:
    """
    Persist validated transactions to the database and save CSV backup.

    Args:
        df (pd.DataFrame): Validated transaction data
    """
    logger.info("Starting transaction load")

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        try:
            df.to_sql(
                "transactions",
                conn,
                if_exists="append",
                index=False
            )
            logger.info(f"Loaded {len(df)} transactions to database")

        except sqlite3.IntegrityError as e:
            logger.warning("Duplicate transaction detected, skipped insertion")
            logger.debug(str(e))

        except Exception as e:
            logger.error("Failed to load transactions")
            raise

    # Save to CSV backup
    save_processed_csv(df)

