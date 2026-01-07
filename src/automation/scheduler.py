"""
Job scheduler for FinOpsia automation tasks.
Uses APScheduler for cron-like scheduling.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

scheduler = BackgroundScheduler()


def add_cron_job(func, cron: dict, job_id: str, **kwargs):
    """
    Add a cron job to the scheduler.
    :param func: function to schedule
    :param cron: dict with cron params (minute, hour, day_of_week, etc)
    :param job_id: unique job id
    :param kwargs: extra args for func
    """
    trigger = CronTrigger(**cron)
    scheduler.add_job(func, trigger, id=job_id, kwargs=kwargs, replace_existing=True)
    logger.info(f"Scheduled job '{job_id}' with cron: {cron}")


def start_scheduler():
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started.")


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
