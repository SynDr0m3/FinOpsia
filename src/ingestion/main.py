from .reader import read_transactions
from .validator import validate_transactions
from .loader import load_transactions
from monitoring.logger import logger
from pathlib import Path


DATA_PATH = Path("data/raw/transactions.csv")



def run_ingestion(csv_path: Path = DATA_PATH, user_id: str = "test_user"):
    """Full ingestion pipeline: read → validate → persist"""
    logger.info("Starting full ingestion pipeline", extra={"account_id": None, "user_id": user_id})

    # Step 1: Read
    df = read_transactions(csv_path)
    logger.info(f"Read {len(df)} transactions from CSV", extra={"account_id": None, "user_id": user_id})

    # Step 2: Validate
    df = validate_transactions(df)
    logger.info(f"{len(df)} transactions after validation", extra={"account_id": None, "user_id": user_id})

    # Step 3: Load (DB + processed CSV)
    load_transactions(df, user_id=user_id)
    logger.info("Ingestion pipeline complete", extra={"account_id": None, "user_id": user_id})


if __name__ == "__main__":
    run_ingestion(user_id="test_user")
