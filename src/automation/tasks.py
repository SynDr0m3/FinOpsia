"""
Defines automated tasks for FinOpsia platform.
Each task can be scheduled via the scheduler.
"""
from loguru import logger
from automation.retry import retry


# Real pipeline imports
from ingestion.main import run_ingestion
from db.repositories import _get_connection
from ml.persistence import get_model
from ml.forecaster import forecast_balance

def get_all_account_ids():
    """Fetch all account_ids from the database."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT account_id FROM transactions")
        rows = cursor.fetchall()
    return [row[0] for row in rows]


@retry(max_attempts=3, delay=5)
def ingest_transactions():
    """
    Ingest new transactions from data source.
    Should be idempotent and safe if no new data.
    """
    logger.info("[TASK] Running transaction ingestion...")
    try:
        run_ingestion()
        logger.info("[TASK] Ingestion complete.")
    except Exception as e:
        logger.error(f"[TASK] Ingestion failed: {e}")


@retry(max_attempts=3, delay=10)
def retrain_categorizer():
    """
    Retrain the categorizer model (scheduled or manual).
    """
    logger.info("[TASK] Retraining categorizer model...")
    # TODO: Implement retraining logic if needed
    logger.info("[TASK] Categorizer retraining complete.")


@retry(max_attempts=3, delay=10)
def retrain_forecasters():
    """
    Retrain all forecaster models (scheduled or manual).
    """
    logger.info("[TASK] Retraining all forecaster models...")
    for account_id in get_all_account_ids():
        try:
            model = get_model("forecaster", account_id=account_id)
            logger.info(f"[TASK] Forecaster retrained for account {account_id}")
        except Exception as e:
            logger.error(f"[TASK] Forecaster retraining failed for account {account_id}: {e}")
    logger.info("[TASK] Forecaster retraining complete.")


@retry(max_attempts=3, delay=5)
def run_forecasts():
    """
    Generate forecasts for all accounts.
    """
    logger.info("[TASK] Running forecasts for all accounts...")
    for account_id in get_all_account_ids():
        try:
            model = get_model("forecaster", account_id=account_id)
            forecast = forecast_balance(model, days_ahead=7)
            logger.info(f"[TASK] Forecast for account {account_id}:\n{forecast}")
        except Exception as e:
            logger.error(f"[TASK] Forecasting failed for account {account_id}: {e}")
    logger.info("[TASK] Forecasting complete.")
