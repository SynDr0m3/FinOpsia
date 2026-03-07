"""
Defines automated tasks for FinOpsia platform.
Each task can be scheduled via the scheduler.
"""
from src.monitoring.logger import logger
from src.automation.retry import retry


# Real pipeline imports
from src.ingestion.main import run_ingestion
from src.db.repositories import fetch_user_account_ids
from src.ml.persistence import get_model, train_and_save_model
from src.ml.forecaster import forecast_balance


def get_all_account_ids(user_id: str):
    """Fetch all account_ids for a user from the database."""
    return fetch_user_account_ids(user_id)


@retry(max_attempts=3, delay=5)
def ingest_transactions(user_id: int | str = 1):
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
            train_and_save_model(
                model_type="forecaster",
                account_id=account_id,
                user_id=user_id,
            )
            logger.info(f"[TASK] Retrained forecaster for account {account_id} (user {user_id})")
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
            model = get_model(
                model_type="forecaster",
                account_id=account_id,
                user_id=user_id,
            )
            forecast = forecast_balance(model, days_ahead=7)
            logger.info(f"[TASK] Generated {len(forecast)} forecast rows for account {account_id} (user {user_id})")
        except Exception as e:
            logger.error(f"[TASK] Forecasting failed for account {account_id} (user {user_id}): {e}")
    logger.info(f"[TASK] Forecasting complete for user {user_id}.")
