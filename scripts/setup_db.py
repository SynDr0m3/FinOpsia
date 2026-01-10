"""
FinOpsia Database Setup Script.

Creates the SQLite database with all required tables:
    - transactions: Stores categorized transaction data
    - account_metadata: Stores account-level info for forecasting

Run this script once before using the pipeline:
    python scripts/setup_db.py
"""

import sqlite3
from pathlib import Path

# Database path (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "finopsia.db"


def setup_database():
    """Create database and all required tables."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # =========================================================
    # 1. Users Table
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)
    print("OK users table created")

    # =========================================================
    # 2. Accounts Table (linked to users)
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            account_number TEXT UNIQUE NOT NULL CHECK(length(account_number) = 10 AND account_number GLOB '[0-9]*'),
            starting_balance INTEGER NOT NULL,
            currency TEXT NOT NULL,
            last_updated DATE,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    print("OK accounts table created")

    # =========================================================
    # 3. Transactions Table (from loader.py)
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            amount INTEGER NOT NULL,
            direction TEXT CHECK(direction IN ('inflow', 'outflow')),
            currency TEXT,
            transaction_date DATE NOT NULL,
            posted_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    """)
    print("OK transactions table created")

    conn.commit()
    conn.close()

    print(f"\nOK Database created at: {DB_PATH.absolute()}")


def insert_sample_users_and_accounts():
    """Insert sample users and accounts for testing."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Sample users
    sample_users = [
        ("alice", "hash1", "alice@example.com"),
        ("bob", "hash2", "bob@example.com"),
        ("carol", "hash3", "carol@example.com"),
        ("dave", "hash4", "dave@example.com"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO users (username, password_hash, email)
        VALUES (?, ?, ?)
    """, sample_users)

    # Get user_ids for linking
    cur.execute("SELECT user_id, username FROM users")
    user_map = {row[1]: row[0] for row in cur.fetchall()}

    # Sample accounts (account_id, user_id, account_number, starting_balance, currency, last_updated)
    sample_accounts = [
        ("1", user_map["alice"], "0449371890", 50000000, "NGN", "2025-01-01"),
        ("2", user_map["bob"],   "0449371891", 75000000, "NGN", "2025-01-01"),
        ("3", user_map["carol"], "0449371892", 100000000, "NGN", "2025-01-01"),
        ("4", user_map["dave"],  "0449371893", 25000000, "NGN", "2025-01-01"),
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO accounts
        (account_id, user_id, account_number, starting_balance, currency, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_accounts)

    conn.commit()
    conn.close()
    print(f"OK Inserted {len(sample_users)} sample user(s) and {len(sample_accounts)} sample account(s)")


if __name__ == "__main__":
    print("=" * 50)
    print("FinOpsia Database Setup")
    print("=" * 50)

    setup_database()
    insert_sample_users_and_accounts()

    print("\nOK Setup complete!")
