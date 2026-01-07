"""
Retry and recovery logic for automation tasks.
Provides a retry decorator with exponential backoff.
"""
import time
import functools
from loguru import logger


def retry(max_attempts=3, delay=2, backoff=2, exceptions=(Exception,)):
    """
    Decorator to retry a function with exponential backoff.
    :param max_attempts: max retries
    :param delay: initial delay in seconds
    :param backoff: multiplier for delay
    :param exceptions: exceptions to catch
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            wait = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Retry {attempts}/{max_attempts} for {func.__name__} in {wait}s: {e}")
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
