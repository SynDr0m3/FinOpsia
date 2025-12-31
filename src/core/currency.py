"""
Currency utilities.

Handles safe conversion between major currency units
(e.g. naira, dollar) and smallest units (kobo, cents).

IMPORTANT:
- All monetary values stored in the database MUST be
  in the smallest unit to avoid floating point errors.
"""

from loguru import logger
from typing import Dict


# Number of decimal places per currency
CURRENCY_DECIMALS: Dict[str, int] = {
    "NGN": 2,  # kobo
    "USD": 2,  # cents
    "EUR": 2,  # cents
    "JPY": 0,  # no decimals
}


def to_smallest_unit(amount: float, currency: str) -> int:
    """
    Convert a major currency amount to its smallest unit.

    Example:
        1000.50 NGN -> 100050 kobo
    """
    if currency not in CURRENCY_DECIMALS:
        logger.error(f"Unsupported currency: {currency}")
        raise ValueError(f"Unsupported currency: {currency}")

    factor = 10 ** CURRENCY_DECIMALS[currency]
    smallest = int(round(amount * factor))

    logger.debug(
        f"Converted {amount} {currency} -> {smallest} (smallest unit)"
    )
    return smallest


def from_smallest_unit(amount_int: int, currency: str) -> float:
    """
    Convert an amount from smallest unit back to major unit.

    Example:
        100050 kobo -> 1000.50 NGN
    """
    if currency not in CURRENCY_DECIMALS:
        logger.error(f"Unsupported currency: {currency}")
        raise ValueError(f"Unsupported currency: {currency}")

    factor = 10 ** CURRENCY_DECIMALS[currency]
    major = amount_int / factor

    logger.debug(
        f"Converted {amount_int} (smallest unit) -> {major} {currency}"
    )
    return major
