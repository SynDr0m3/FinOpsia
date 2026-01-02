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

# Database path (matches codebase)
DB_PATH = Path("data/finopsia.db")


def setup_database():
    """Create database and all required tables."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # =========================================================
    # 1. Transactions Table (from loader.py)
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            amount INTEGER NOT NULL,
            direction TEXT CHECK(direction IN ('inflow', 'outflow')),
            transaction_date DATE NOT NULL,
            posted_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    print("✓ transactions table created")

    # =========================================================
    # 2. Account Metadata Table (from repositories.py)
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS account_metadata (
            account_id TEXT PRIMARY KEY,
            starting_balance INTEGER NOT NULL,
            currency TEXT NOT NULL,
            last_updated DATE
        )
    """)

    print("✓ account_metadata table created")

    conn.commit()
    conn.close()

    print(f"\n✅ Database created at: {DB_PATH.absolute()}")


def insert_sample_accounts():
    """Insert sample account metadata for testing."""

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Sample accounts (matching account_ids in transactions.csv)
    sample_accounts = [
        ("1", 50000000, "NGN", "2025-01-01"),
        ("2", 75000000, "NGN", "2025-01-01"),
        ("3", 100000000, "NGN", "2025-01-01"),
        ("4", 25000000, "NGN", "2025-01-01"),
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO account_metadata
        (account_id, starting_balance, currency, last_updated)
        VALUES (?, ?, ?, ?)
    """, sample_accounts)

    conn.commit()
    conn.close()

    print(f"✓ Inserted {len(sample_accounts)} sample account(s)")


if __name__ == "__main__":
    print("=" * 50)
    print("FinOpsia Database Setup")
    print("=" * 50)

    setup_database()
    insert_sample_accounts()

    print("\n✅ Setup complete!")
