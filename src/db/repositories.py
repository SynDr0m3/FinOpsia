"""
Database repository functions.

These functions provide a controlled way to read data
from the database without exposing raw SQL everywhere.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict
from src.monitoring.logger import logger
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
        conn.execute("PRAGMA foreign_keys = ON")
        logger.debug("Database connection established successfully", extra={"account_id": None, "user_id": None})
        return conn

    except sqlite3.Error as exc:
        logger.exception(
            f"Failed to connect to database at {DB_PATH}", extra={"account_id": None, "user_id": None}
        )
        raise


def _normalize_user_id(user_id: int | str | None) -> int | None:
    """Normalize user identifiers from API/CLI inputs."""
    if user_id is None:
        return None
    if isinstance(user_id, int):
        return user_id
    if isinstance(user_id, str) and user_id.isdigit():
        return int(user_id)
    return None


def fetch_account_owner_id(account_id: str) -> int:
    """Return the owner user_id for an account."""
    logger.debug(f"Resolving owner for account_id={account_id}", extra={"account_id": account_id, "user_id": None})

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_id
            FROM accounts
            WHERE account_id = ?
            """,
            (account_id,),
        )
        row = cursor.fetchone()

    if row is None:
        logger.error(f"Account owner not found for {account_id}", extra={"account_id": account_id, "user_id": None})
        raise ValueError(f"Account {account_id} does not exist")

    return row[0]



def fetch_account_metadata(account_id: str, user_id: str | None = None) -> Dict:
    """
    Fetch account metadata such as starting balance and currency.
    Log user_id for traceability.
    """
    logger.info(f"Fetching account metadata for account_id={account_id}", extra={"account_id": account_id, "user_id": user_id})

    normalized_user_id = _normalize_user_id(user_id)
    owner_id = fetch_account_owner_id(account_id)
    if normalized_user_id is not None and owner_id != normalized_user_id:
        logger.warning(
            f"User {normalized_user_id} attempted to access account {account_id}",
            extra={"account_id": account_id, "user_id": normalized_user_id},
        )
        raise PermissionError(f"User {normalized_user_id} does not own account {account_id}")

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, account_number, starting_balance, currency, last_updated
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
        "account_number": row[1],
        "starting_balance": row[2],
        "currency": row[3],
        "last_updated": row[4],
    }

    logger.debug(f"Account metadata fetched: {result}", extra={"account_id": account_id, "user_id": user_id})
    return result


def fetch_transactions(
    account_id: str,
    years_back: int | None = None,
    user_id: int | str | None = None,
) -> pd.DataFrame:
    """
    Fetch transactions for a given account.
    Log account_id for traceability.
    """
    logger.info(f"Fetching transactions for account_id={account_id}", extra={"account_id": account_id, "user_id": None})

    normalized_user_id = _normalize_user_id(user_id)
    owner_id = fetch_account_owner_id(account_id)
    if normalized_user_id is not None and owner_id != normalized_user_id:
        logger.warning(
            f"User {normalized_user_id} attempted to access transactions for account {account_id}",
            extra={"account_id": account_id, "user_id": normalized_user_id},
        )
        raise PermissionError(f"User {normalized_user_id} does not own account {account_id}")

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


def fetch_user_by_username(username: str) -> Dict | None:
    """
    Fetch user by username for authentication.
    
    Returns:
    - Dict with user_id, username, email, password_hash if found
    - None if user does not exist
    """
    logger.info(f"Fetching user by username: {username}", extra={"account_id": None, "user_id": None})

    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT user_id, username, email, password_hash, role
                FROM users
                WHERE username = ?
                """,
                (username,),
            )
            row = cursor.fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column: role" not in str(exc):
                raise
            cursor.execute(
                """
                SELECT user_id, username, email, password_hash
                FROM users
                WHERE username = ?
                """,
                (username,),
            )
            row = cursor.fetchone()
    
    if row is None:
        logger.debug(f"User not found: {username}", extra={"account_id": None, "user_id": None})
        return None
    
    result = {
        "user_id": row[0],
        "username": row[1],
        "email": row[2],
        "hashed_password": row[3],
        "role": row[4] if len(row) > 4 else "user",
    }
    
    logger.debug(f"User fetched: {username}", extra={"account_id": None, "user_id": None})
    return result


def fetch_user_by_id(user_id: int) -> Dict | None:
    """
    Fetch user by ID.
    
    Returns:
    - Dict with user_id, username, email if found
    - None if user does not exist
    """
    logger.info(f"Fetching user by ID: {user_id}", extra={"account_id": None, "user_id": user_id})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT user_id, username, email, role
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
        except sqlite3.OperationalError as exc:
            if "no such column: role" not in str(exc):
                raise
            cursor.execute(
                """
                SELECT user_id, username, email
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
    
    if row is None:
        logger.error(f"User not found: {user_id}", extra={"account_id": None, "user_id": user_id})
        return None
    
    result = {
        "user_id": row[0],
        "username": row[1],
        "email": row[2],
        "role": row[3] if len(row) > 3 else "user",
    }
    
    return result


def fetch_user_accounts(user_id: int) -> List[Dict]:
    """
    Fetch all accounts for a user.
    
    Returns:
    - List of dicts with account_id, account_number, starting_balance, currency, last_updated
    """
    logger.info(f"Fetching accounts for user_id={user_id}", extra={"account_id": None, "user_id": user_id})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, account_number, starting_balance, currency, last_updated
            FROM accounts
            WHERE user_id = ?
            ORDER BY account_id
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    
    result = [
        {
            "account_id": row[0],
            "account_number": row[1],
            "starting_balance": row[2],
            "currency": row[3],
            "last_updated": row[4],
        }
        for row in rows
    ]
    
    logger.debug(f"Fetched {len(result)} accounts for user {user_id}", extra={"account_id": None, "user_id": user_id})
    return result


def fetch_user_account_ids(user_id: int | str) -> List[str]:
    """Return all account IDs owned by a user."""
    normalized_user_id = _normalize_user_id(user_id)
    if normalized_user_id is None:
        raise ValueError("user_id must be numeric")
    return [account["account_id"] for account in fetch_user_accounts(normalized_user_id)]


def fetch_all_transactions(
    account_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> List[Dict]:
    """
    Fetch all transactions for an account with optional date filtering.
    
    Args:
    - account_id: Account to fetch transactions for
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    
    Returns:
    - List of transaction dicts with full details
    """
    logger.info(f"Fetching all transactions for account_id={account_id}", extra={"account_id": account_id, "user_id": None})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic query with date filters
        query = """
            SELECT transaction_id, account_id, description, category, 
                   amount, direction, transaction_date, posted_at
            FROM transactions
            WHERE account_id = ?
        """
        params = [account_id]
        
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY transaction_date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    result = [
        {
            "transaction_id": row[0],
            "account_id": row[1],
            "description": row[2],
            "category": row[3],
            "amount": row[4],
            "direction": row[5],
            "transaction_date": row[6],
            "posted_at": row[7],
        }
        for row in rows
    ]
    
    logger.debug(f"Fetched {len(result)} transactions", extra={"account_id": account_id, "user_id": None})
    return result


def verify_account_ownership(account_id: str, user_id: int) -> bool:
    """
    Verify that a user owns a specific account.
    
    Returns:
    - True if user owns the account
    - False otherwise
    """
    logger.debug(f"Verifying ownership: user_id={user_id}, account_id={account_id}", extra={"account_id": account_id, "user_id": user_id})
    
    normalized_user_id = _normalize_user_id(user_id)
    if normalized_user_id is None:
        return False

    try:
        owner_id = fetch_account_owner_id(account_id)
    except ValueError:
        logger.warning(f"Account not found: {account_id}", extra={"account_id": account_id, "user_id": user_id})
        return False

    is_owner = owner_id == normalized_user_id
    logger.debug(f"Ownership verified: {is_owner}", extra={"account_id": account_id, "user_id": user_id})
    return is_owner


def create_account(
    account_id: str,
    user_id: int,
    account_number: str,
    starting_balance: int,
    currency: str,
) -> Dict:
    """
    Create a new account for a user.
    
    Returns:
    - Dict with created account details
    """
    logger.info(f"Creating account for user_id={user_id}", extra={"account_id": account_id, "user_id": user_id})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO accounts 
                (account_id, user_id, account_number, starting_balance, currency, last_updated)
                VALUES (?, ?, ?, ?, ?, DATE('now'))
                """,
                (account_id, user_id, account_number, starting_balance, currency),
            )
            conn.commit()
            logger.info(f"Account created: {account_id}", extra={"account_id": account_id, "user_id": user_id})
        except sqlite3.IntegrityError as e:
            logger.error(f"Duplicate account: {account_id}", extra={"account_id": account_id, "user_id": user_id})
            raise ValueError(f"Account {account_id} or account_number {account_number} already exists")
    
    return {
        "account_id": account_id,
        "account_number": account_number,
        "starting_balance": starting_balance,
        "currency": currency,
        "last_updated": str(pd.Timestamp.now().date()),
    }


def update_account(
    account_id: str,
    **kwargs
) -> Dict:
    """
    Update account metadata.
    
    Args:
    - account_id: Account to update
    - **kwargs: Fields to update (starting_balance, currency, etc)
    
    Returns:
    - Updated account dict
    """
    logger.info(f"Updating account: {account_id}", extra={"account_id": account_id, "user_id": None})
    
    # Get current account
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT account_id, account_number, starting_balance, currency, user_id
            FROM accounts
            WHERE account_id = ?
            """,
            (account_id,),
        )
        row = cursor.fetchone()
    
    if row is None:
        raise ValueError(f"Account {account_id} does not exist")
    
    current = {
        "account_id": row[0],
        "account_number": row[1],
        "starting_balance": row[2],
        "currency": row[3],
        "user_id": row[4],
        "last_updated": None,
    }
    
    # Update specified fields
    current.update(kwargs)
    
    # Persist to database
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE accounts
            SET starting_balance = ?, currency = ?, last_updated = DATE('now')
            WHERE account_id = ?
            """,
            (current["starting_balance"], current["currency"], account_id),
        )
        conn.commit()
        cursor.execute(
            """
            SELECT last_updated
            FROM accounts
            WHERE account_id = ?
            """,
            (account_id,),
        )
        updated_row = cursor.fetchone()
        current["last_updated"] = updated_row[0] if updated_row else None
        logger.info(f"Account updated: {account_id}", extra={"account_id": account_id, "user_id": current["user_id"]})
    
    return current


def delete_account(account_id: str) -> bool:
    """
    Delete an account (soft delete via archiving recommended, but this does hard delete).
    
    Returns:
    - True if deleted, False if not found
    """
    logger.warning(f"Deleting account: {account_id}", extra={"account_id": account_id, "user_id": None})
    
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM accounts WHERE account_id = ?", (account_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
    
    if deleted:
        logger.info(f"Account deleted: {account_id}", extra={"account_id": account_id, "user_id": None})
    else:
        logger.warning(f"Account not found for deletion: {account_id}", extra={"account_id": account_id, "user_id": None})
    
    return deleted
