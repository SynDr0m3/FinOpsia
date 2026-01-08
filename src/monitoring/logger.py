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

# Log format
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Remove default loguru handler
logger.remove()

# Add console handler
logger.add(sys.stdout, level=LOG_LEVEL, format=LOG_FORMAT, enqueue=True, backtrace=True, diagnose=True)

# Optional: Add file handler with rotation and retention
LOG_FILE = os.environ.get("FINOPSIA_LOG_FILE", "logs/finopsia.log")
logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# Usage: from monitoring.logger import logger
