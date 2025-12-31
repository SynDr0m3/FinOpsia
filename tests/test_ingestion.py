import pytest
import pandas as pd
from pathlib import Path
from ingestion.main import run_ingestion
from ingestion.reader import read_transactions
from ingestion.validator import validate_transactions
from ingestion.loader import DB_PATH


DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "raw/test_transactions.csv"
PROCESSED_CSV = DATA_DIR / "processed/transactions_processed.csv"


@pytest.fixture
def sample_csv():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CSV.write_text(
        """transaction_id,account_id,description,category,amount,direction,transaction_date,posted_at
        tx1,acc1,Coffee,Food,3.5,outflow,2025-01-01,2025-01-01 08:00
        tx2,acc1,Salary,Income,1000.0,inflow,2025-01-01,2025-01-01 09:00"""
    )
    return RAW_CSV


def test_full_ingestion(sample_csv):
    # Run full ingestion pipeline
    run_ingestion(sample_csv)

    # Read back from DB
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        df_db = pd.read_sql("SELECT * FROM transactions", conn)

    assert not df_db.empty
    assert df_db.shape[0] == 2
    assert "transaction_id" in df_db.columns

    # Check processed CSV exists
    assert PROCESSED_CSV.exists()
