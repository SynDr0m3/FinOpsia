"""
Database repository functions.

These functions provide a controlled way to read data
from the database without exposing raw SQL everywhere.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict
from loguru import logger
import pandas as pd


# Path to SQLite database
DB_PATH = Path("data/finopsia.db")


def _get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Logs connection attempts and failures for observability.
    """
    logger.debug(f"Attempting database connection: {DB_PATH}")

    try:
        conn = sqlite3.connect(DB_PATH)
        logger.debug("Database connection established successfully")
        return conn

    except sqlite3.Error as exc:
        logger.exception(
            f"Failed to connect to database at {DB_PATH}"
        )
        raise


def fetch_account_metadata(account_id: str) -> Dict:
    """
    Fetch account metadata such as starting balance and currency.
    """
    logger.info(f"Fetching account metadata for account_id={account_id}")

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, starting_balance, currency, last_updated
            FROM account_metadata
            WHERE account_id = ?
            """,
            (account_id,),
        )
        row = cursor.fetchone()

    if row is None:
        logger.error(f"Account metadata not found for {account_id}")
        raise ValueError(f"Account {account_id} does not exist")

    result = {
        "account_id": row[0],
        "starting_balance": row[1],
        "currency": row[2],
        "last_updated": row[3],
    }

    logger.debug(f"Account metadata fetched: {result}")
    return result

def fetch_transactions(
    account_id: str,
    years_back: int | None = None,
    ) -> pd.DataFrame:
    logger.info(f"Fetching transactions for account_id={account_id}")

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT transaction_date AS txn_date, amount, direction, category
            FROM transactions
            WHERE account_id = ?
            ORDER BY txn_date
            """,
            (account_id,),
        )
        rows = cursor.fetchall()

    df = pd.DataFrame(
        rows,
        columns=["txn_date", "amount", "direction", "category"],
    )
    df["txn_date"] = pd.to_datetime(df["txn_date"])

    if years_back:
        # Use timezone-naive timestamp to match txn_date
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years_back)
        df = df[df["txn_date"] >= cutoff]

    logger.debug(f"Fetched {len(df)} transactions")
    return df
