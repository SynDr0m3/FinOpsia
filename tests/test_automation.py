"""
Test file for FinOpsia automation module.
Verifies that automation tasks run and scheduler can trigger jobs.
"""
import pytest
from automation.tasks import ingest_transactions, retrain_forecasters, run_forecasts
from automation.scheduler import add_cron_job, start_scheduler, shutdown_scheduler
from apscheduler.schedulers.background import BackgroundScheduler


def test_ingest_transactions_runs():
    """Test that ingestion task runs without error."""
    ingest_transactions()


def test_retrain_forecasters_runs():
    """Test that forecaster retraining runs without error."""
    retrain_forecasters()


def test_run_forecasts_runs():
    """Test that forecasting runs without error."""
    run_forecasts()


def test_scheduler_add_and_run_job():
    """Test that scheduler can add and run a job."""
    scheduler = BackgroundScheduler()
    job_executed = []

    def dummy_job():
        job_executed.append(True)

    scheduler.add_job(dummy_job, 'interval', seconds=1, id='test_job')
    scheduler.start()
    import time
    time.sleep(2)
    scheduler.shutdown()
    assert job_executed, "Scheduler did not execute the job"
