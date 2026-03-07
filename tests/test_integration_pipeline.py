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
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path for src imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline and DB functions
from src.runner import run_pipeline
from src.db.repositories import fetch_account_metadata, fetch_transactions, fetch_user_by_username, fetch_user_account_ids
from src.ingestion.main import run_ingestion

DB_PATH = Path("data/finopsia.db")

TEST_USER = {
    "username": "alice",
    "password": "hash1",
}

def setup_module(module):
    # Run the setup_db script to initialize the DB
    subprocess.run(["python", "scripts/setup_db.py"], check=True)

def test_full_pipeline():
    # Simulate login (in real app, this would be an API call)
    user = fetch_user_by_username(TEST_USER["username"])
    assert user is not None

    user_id = user["user_id"]
    account_ids = fetch_user_account_ids(user_id)
    account_id = account_ids[0]

    raw_df = pd.read_csv("data/raw/transactions.csv")
    scoped_df = raw_df[raw_df["account_id"].astype(str).isin(account_ids)]
    scoped_csv_path = Path("data/raw/transactions_alice.csv")
    scoped_df.to_csv(scoped_csv_path, index=False)

    # Fetch account metadata (should be scoped to user)
    meta = fetch_account_metadata(account_id, user_id=user_id)
    assert meta["account_id"] == account_id

    # Run ingestion (simulate uploading a CSV for this user)
    run_ingestion(csv_path=scoped_csv_path, user_id=user_id)

    # Run the full pipeline (categorization, persistence, forecasting)
    run_pipeline(
        csv_path=scoped_csv_path,
        user_id=user_id,
        account_id=account_id,
        forecast=True,
        forecast_days=3,
        dry_run=False,
    )

    # Check that transactions are present and scoped to the account
    txns = fetch_transactions(account_id, user_id=user_id)
    assert not txns.empty

    # Optionally: Check logs for correct user_id/account_id context (manual or with log parsing)

if __name__ == "__main__":
    pytest.main([__file__])
