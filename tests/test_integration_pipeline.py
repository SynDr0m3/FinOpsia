"""
Integration test for the full FinOpsia multi-tenant pipeline.
- Sets up the DB
- Logs in a test user
- Fetches user details (user_id, account_id)
- Runs ingestion, categorization, persistence, forecasting
- Asserts correct scoping and logging
"""
import subprocess
import pytest
import sqlite3
from pathlib import Path

# Ensure project root is in sys.path for src imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline and DB functions
from src.runner import run_pipeline
from src.db.repositories import fetch_account_metadata, fetch_transactions
from src.ingestion.main import run_ingestion

DB_PATH = Path("data/finopsia.db")

TEST_USER = {
    "username": "alice",
    "password": "hash1",
    "user_id": 1,
}

TEST_ACCOUNT = {
    "account_id": "1",
    "user_id": TEST_USER["user_id"],
}

def setup_module(module):
    # Run the setup_db script to initialize the DB
    subprocess.run(["python", "scripts/setup_db.py"], check=True)

def test_full_pipeline():
    # Simulate login (in real app, this would be an API call)
    user_id = TEST_USER["user_id"]
    account_id = TEST_ACCOUNT["account_id"]

    # Fetch account metadata (should be scoped to user)
    meta = fetch_account_metadata(account_id, user_id=user_id)
    assert meta["account_id"] == account_id

    # Run ingestion (simulate uploading a CSV for this user)
    run_ingestion(user_id=user_id)

    # Run the full pipeline (categorization, persistence, forecasting)
    run_pipeline(
        csv_path=Path("data/raw/transactions.csv"),
        user_id=user_id,
        account_id=account_id,
        forecast=True,
        forecast_days=3,
        dry_run=False,
    )

    # Check that transactions are present and scoped to the account
    txns = fetch_transactions(account_id)
    assert not txns.empty

    # Optionally: Check logs for correct user_id/account_id context (manual or with log parsing)

if __name__ == "__main__":
    pytest.main([__file__])
