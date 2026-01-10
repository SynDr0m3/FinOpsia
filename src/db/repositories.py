"""
Database repository functions.

These functions provide a controlled way to read data
from the database without exposing raw SQL everywhere.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict
from monitoring.logger import logger
import pandas as pd


# Path to SQLite database
DB_PATH = Path("data/finopsia.db")


def _get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Logs connection attempts and failures for observability.
    """
    logger.debug(f"Attempting database connection: {DB_PATH}", extra={"account_id": None, "user_id": None})

    try:
        conn = sqlite3.connect(DB_PATH)
        logger.debug("Database connection established successfully", extra={"account_id": None, "user_id": None})
        return conn

    except sqlite3.Error as exc:
        logger.exception(
            f"Failed to connect to database at {DB_PATH}", extra={"account_id": None, "user_id": None}
        )
        raise



def fetch_account_metadata(account_id: str, user_id: str | None = None) -> Dict:
    """
    Fetch account metadata such as starting balance and currency.
    Log user_id for traceability.
    """
    logger.info(f"Fetching account metadata for account_id={account_id}", extra={"account_id": account_id, "user_id": user_id})

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, starting_balance, currency, last_updated
            FROM accounts
            WHERE account_id = ?
            """,
            (account_id,),
        )
        row = cursor.fetchone()

    if row is None:
        logger.error(f"Account metadata not found for {account_id}", extra={"account_id": account_id, "user_id": user_id})
        raise ValueError(f"Account {account_id} does not exist")

    result = {
        "account_id": row[0],
        "starting_balance": row[1],
        "currency": row[2],
        "last_updated": row[3],
    }

    logger.debug(f"Account metadata fetched: {result}", extra={"account_id": account_id, "user_id": user_id})
    return result


def fetch_transactions(
    account_id: str,
    years_back: int | None = None,
) -> pd.DataFrame:
    """
    Fetch transactions for a given account.
    Log account_id for traceability.
    """
    logger.info(f"Fetching transactions for account_id={account_id}", extra={"account_id": account_id, "user_id": None})

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

    logger.debug(f"Fetched {len(df)} transactions", extra={"account_id": account_id, "user_id": None})
    return df
