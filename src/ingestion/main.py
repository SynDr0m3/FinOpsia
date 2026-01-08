from .reader import read_transactions
from .validator import validate_transactions
from .loader import load_transactions
from monitoring.logger import logger
from pathlib import Path


DATA_PATH = Path("data/raw/transactions.csv")


def run_ingestion(csv_path: Path = DATA_PATH):
    """Full ingestion pipeline: read → validate → persist"""
    logger.info("Starting full ingestion pipeline")

    # Step 1: Read
    df = read_transactions(csv_path)
    logger.info(f"Read {len(df)} transactions from CSV")

    # Step 2: Validate
    df = validate_transactions(df)
    logger.info(f"{len(df)} transactions after validation")

    # Step 3: Load (DB + processed CSV)
    load_transactions(df)
    logger.info("Ingestion pipeline complete")


if __name__ == "__main__":
    run_ingestion()
