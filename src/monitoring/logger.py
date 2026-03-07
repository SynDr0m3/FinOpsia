"""
Centralized logging configuration for FinOpsia.
Sets up loguru with preferred format, level, and handlers.
Import this module at the top of your main scripts to initialize logging.
"""
from loguru import logger
import sys
import os
from pathlib import Path

# Log level (can be set via env or default to INFO)
LOG_LEVEL = os.environ.get("FINOPSIA_LOG_LEVEL", "INFO")
LOG_ENQUEUE = os.environ.get("FINOPSIA_LOG_ENQUEUE", "").lower() in {"1", "true", "yes"}


# Log format (human-readable, safe for Loguru)
LOG_FORMAT = "{level.name} | {name}:{function}:{line} - {message}"

# Remove default loguru handler
logger.remove()

# Add console handler (human-readable)
logger.add(sys.stdout, level=LOG_LEVEL, format=LOG_FORMAT, enqueue=LOG_ENQUEUE, backtrace=True, diagnose=True)

# Optional: Add file handler with rotation and retention (JSON format)
LOG_FILE = os.environ.get("FINOPSIA_LOG_FILE", "logs/finopsia.log")
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    serialize=True,
    enqueue=LOG_ENQUEUE,
    backtrace=True,
    diagnose=True,
)

