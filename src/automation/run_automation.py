"""
Entrypoint for FinOpsia automation service.
Starts the scheduler and schedules real pipeline jobs.
"""
from automation.scheduler import add_cron_job, start_scheduler
from automation.tasks import ingest_transactions, retrain_forecasters, run_forecasts
from monitoring.logger import logger
import time

if __name__ == "__main__":
    logger.info("Starting FinOpsia automation service...")

    # Schedule jobs (real pipeline)
    # Daily ingestion at 6:00 AM
    add_cron_job(ingest_transactions, {"hour": 6, "minute": 0}, job_id="daily_ingestion")

    # Weekly forecaster retraining at Sunday midnight
    add_cron_job(retrain_forecasters, {"day_of_week": "sun", "hour": 0, "minute": 0}, job_id="weekly_forecaster_retrain")

    # Daily forecasts at 7:00 AM
    add_cron_job(run_forecasts, {"hour": 7, "minute": 0}, job_id="daily_forecast")

    start_scheduler()
    logger.info("Scheduler running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Automation service stopped.")
