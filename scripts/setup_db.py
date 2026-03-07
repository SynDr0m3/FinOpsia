"""
FinOpsia Database Setup Script.

Creates the SQLite database with all required tables and seeds
deterministic dummy credentials for local testing.

Run this script once before using the pipeline:
    python scripts/setup_db.py
"""

import base64
import hashlib
import hmac
import pandas as pd
import secrets
import sqlite3
from pathlib import Path

# Database path (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "finopsia.db"
PBKDF2_ITERATIONS = 390000


def hash_password(password: str) -> str:
    """Hash password for seeded login accounts without external dependencies."""
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    encoded_salt = base64.b64encode(salt).decode("ascii")
    encoded_hash = base64.b64encode(derived_key).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${encoded_salt}${encoded_hash}"


def ensure_users_role_column(cur: sqlite3.Cursor) -> None:
    """Backfill the role column for older local databases."""
    cur.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cur.fetchall()}
    if "role" not in columns:
        cur.execute(
            "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user' "
            "CHECK(role IN ('admin', 'user'))"
        )
        print("OK users.role column added")


def prepare_transaction_seed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw transaction CSV into the DB storage format."""
    prepared = df.copy()
    prepared["account_id"] = prepared["account_id"].astype(str)
    prepared["transaction_id"] = prepared["transaction_id"].astype(str)
    prepared["transaction_date"] = pd.to_datetime(prepared["transaction_date"], errors="raise")
    prepared["posted_at"] = pd.to_datetime(prepared["posted_at"], errors="raise")
    prepared["amount"] = pd.to_numeric(prepared["amount"], errors="raise").mul(100).round().astype(int)
    prepared["direction"] = prepared["direction"].str.lower().str.strip()
    prepared["currency"] = prepared.get("currency", "NGN")
    return prepared[
        [
            "transaction_id",
            "account_id",
            "description",
            "category",
            "amount",
            "direction",
            "currency",
            "transaction_date",
            "posted_at",
        ]
    ]


def setup_database():
    """Create database and all required tables."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # =========================================================
    # 1. Users Table
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user'))
        )
    """)
    ensure_users_role_column(cur)
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
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # Clear previous demo records so reruns stay deterministic.
    cur.execute("DELETE FROM transactions WHERE account_id IN ('1', '2', '3', '4')")
    cur.execute("DELETE FROM accounts WHERE account_id IN ('1', '2', '3', '4')")
    cur.execute(
        "DELETE FROM users WHERE username IN ('admin', 'alice', 'bob', 'carol', 'dave')"
    )
    cur.execute("DELETE FROM sqlite_sequence WHERE name = 'users'")

    # Sample users
    sample_users = [
        (1, "admin", hash_password("AdminPass123!"), "admin@finopsia.local", "admin"),
        (2, "alice", hash_password("AlicePass123!"), "alice@example.com", "user"),
        (3, "bob", hash_password("BobPass123!"), "bob@example.com", "user"),
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO users (user_id, username, password_hash, email, role)
        VALUES (?, ?, ?, ?, ?)
    """, sample_users)

    # Get user_ids for linking
    cur.execute("SELECT user_id, username FROM users")
    user_map = {row[1]: row[0] for row in cur.fetchall()}

    # Sample accounts (account_id, user_id, account_number, starting_balance, currency, last_updated)
    sample_accounts = [
        ("1", user_map["alice"], "0449371890", 50000000, "NGN", "2025-01-01"),
        ("2", user_map["alice"], "0449371891", 75000000, "NGN", "2025-01-01"),
        ("3", user_map["bob"], "0449371892", 100000000, "NGN", "2025-01-01"),
        ("4", user_map["bob"], "0449371893", 25000000, "NGN", "2025-01-01"),
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO accounts
        (account_id, user_id, account_number, starting_balance, currency, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sample_accounts)

    conn.commit()
    conn.close()
    print(f"OK Inserted {len(sample_users)} sample user(s) and {len(sample_accounts)} sample account(s)")
    print("\nSeeded login credentials:")
    print("  admin / AdminPass123!  (admin access)")
    print("  alice / AlicePass123!  (accounts 1, 2)")
    print("  bob   / BobPass123!    (accounts 3, 4)")


def seed_transactions_from_csv():
    """Load demo transactions into the database for accounts 1-4."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "transactions.csv"
    if not csv_path.exists():
        print(f"SKIP transactions seed missing file: {csv_path}")
        return

    df = prepare_transaction_seed_df(pd.read_csv(csv_path))

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions")
    conn.commit()

    df.to_sql("transactions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"OK Seeded {len(df)} transaction(s) from {csv_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("FinOpsia Database Setup")
    print("=" * 50)

    setup_database()
    insert_sample_users_and_accounts()
    seed_transactions_from_csv()

    print("\nOK Setup complete!")
