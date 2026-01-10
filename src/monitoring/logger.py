"""
Centralized logging configuration for FinOpsia.
Sets up loguru with preferred format, level, and handlers.
Import this module at the top of your main scripts to initialize logging.
"""
from loguru import logger
import sys
import os

# Log level (can be set via env or default to INFO)
LOG_LEVEL = os.environ.get("FINOPSIA_LOG_LEVEL", "INFO")


# Log format (human-readable, safe for Loguru)
LOG_FORMAT = "{level.name} | {name}:{function}:{line} - {message}"

# JSON format for structured logging (account_id, user_id, etc.)
LOG_JSON_FORMAT = '{"level":"{level.name}","name":"{name}","function":"{function}","line":{line},"message":"{message}","account_id":"{extra[account_id]}","user_id":"{extra[user_id]}"}'


# Remove default loguru handler
logger.remove()

# Add console handler (human-readable)
logger.add(sys.stdout, level=LOG_LEVEL, format=LOG_FORMAT, enqueue=True, backtrace=True, diagnose=True)

# Optional: Add file handler with rotation and retention (JSON format)
LOG_FILE = os.environ.get("FINOPSIA_LOG_FILE", "logs/finopsia.log")
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    format=LOG_JSON_FORMAT,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

