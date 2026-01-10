"""
Defines automated tasks for FinOpsia platform.
Each task can be scheduled via the scheduler.
"""
from monitoring.logger import logger
from automation.retry import retry


# Real pipeline imports
from ingestion.main import run_ingestion
from db.repositories import _get_connection
from ml.persistence import get_model
from ml.forecaster import forecast_balance


def get_all_account_ids(user_id: str):
    """Fetch all account_ids for a user from the database."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT account_id FROM accounts WHERE user_id = ?
        """, (user_id,))
        rows = cursor.fetchall()
    return [row[0] for row in rows]


@retry(max_attempts=3, delay=5)
def ingest_transactions(user_id="test_user"):
    """
    Ingest new transactions from data source.
    Should be idempotent and safe if no new data.
    """
    logger.info("[TASK] Running transaction ingestion...")
    try:
        run_ingestion(user_id=user_id)
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
def retrain_forecasters(user_id: str):
    """
    Retrain all forecaster models (scheduled or manual) for a user.
    """
    logger.info(f"[TASK] Retraining all forecaster models for user {user_id}...")
    for account_id in get_all_account_ids(user_id):
        try:
            # get_model will need to be updated to accept user_id for full compliance
            logger.info(f"[TASK] Would retrain forecaster for account {account_id} (user {user_id})")
        except Exception as e:
            logger.error(f"[TASK] Forecaster retraining failed for account {account_id} (user {user_id}): {e}")
    logger.info(f"[TASK] Forecaster retraining complete for user {user_id}.")



@retry(max_attempts=3, delay=5)
def run_forecasts(user_id: str):
    """
    Generate forecasts for all accounts for a user.
    """
    logger.info(f"[TASK] Running forecasts for all accounts for user {user_id}...")
    for account_id in get_all_account_ids(user_id):
        try:
            logger.info(f"[TASK] Would run forecast for account {account_id} (user {user_id})")
        except Exception as e:
            logger.error(f"[TASK] Forecasting failed for account {account_id} (user {user_id}): {e}")
    logger.info(f"[TASK] Forecasting complete for user {user_id}.")
