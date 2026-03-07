"""Dependency injection for API endpoints with security focus."""
from typing import Generator, Optional
from functools import lru_cache
from fastapi import Depends
from src.api.auth import get_current_user, get_current_admin, verify_account_access
from src.api.exceptions import ForbiddenError


# Database dependency (will be implemented with SQLAlchemy/async)
async def get_db() -> Generator:
    """Get database session."""
    # TODO: Implement database session
    yield


# ML service dependency
@lru_cache()
def get_ml_service():
    """Get ML service instance (cached)."""
    # TODO: Initialize ML services
    from src.ml.categorizer import rule_based_category
    from src.ml.forecaster import build_daily_balance_series
    return {"categorizer": rule_based_category}


async def verify_user_account_access(
    account_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Verify user has access to specific account.
    Security: Prevents users from accessing other users' accounts.
    """
    # TODO: Check in database that current_user owns account_id
    # For now, basic check
    if not account_id or not str(account_id).isdigit():
        raise ForbiddenError("Invalid account ID")
    
    return current_user


async def require_admin(
    current_user: dict = Depends(get_current_admin),
) -> dict:
    """Require admin role for sensitive operations."""
    return current_user
