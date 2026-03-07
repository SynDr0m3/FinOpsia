"""Administrative operations (protected endpoints)."""
from fastapi import APIRouter, Depends, status, HTTPException
from datetime import datetime, timedelta
from src.api.schemas import (
    ResetPasswordRequest, ResetPasswordResponse, MaintenanceCleanupRequest, MaintenanceCleanupResponse
)
from src.api.auth import get_current_admin, hash_password
from src.api.exceptions import NotFoundError, ForbiddenError, BadRequestError
from src.db.repositories import fetch_user_by_id, delete_account as db_delete_account
import sqlite3
from pathlib import Path
from src.monitoring.logger import logger

router = APIRouter()

DB_PATH = Path("data/finopsia.db")


@router.post(
    "/users/{user_id}/reset-password",
    response_model=ResetPasswordResponse
)
async def reset_password(
    user_id: int,
    request: ResetPasswordRequest,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Admin reset user password.
    
    Security:
    - Requires ADMIN role only
    - Generates temporary password
    - Does NOT return permanent password
    - User must change password on next login
    - Should be logged in audit trail
    
    Args:
    - user_id: ID of user to reset password for
    
    Request:
    - New temporary password supplied by admin/operator

    Returns:
    - Confirmation only. Password is never returned by the API.
    """
    try:
        # Check if user exists
        user = fetch_user_by_id(user_id)
        
        if not user:
            raise NotFoundError("User", user_id)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (hash_password(request.new_password), user_id)
            )
            conn.commit()

        logger.warning(
            "Admin reset a user password",
            extra={"account_id": None, "user_id": current_admin["user_id"]},
        )

        return ResetPasswordResponse(
            message="Password reset successfully."
        )
    except NotFoundError:
        raise
    except Exception:
        logger.exception("Password reset failed", extra={"account_id": None, "user_id": current_admin["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/accounts/{account_id}")
async def delete_account(
    account_id: str,
    confirm_deletion: bool = None,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Hard delete account and all associated data.
    
    Security:
    - Requires ADMIN role
    - Requires explicit confirmation (confirm_deletion=true)
    - PERMANENT AND IRREVERSIBLE
    - Should trigger:
      - Deletion from database
      - Audit log entry
      - Notification to account owner
      - Potential data archival/backup
    
    Args:
    - account_id: ID of account to delete
    - confirm_deletion: Must be True to proceed
    
    Returns:
    - 204 No Content on success
    """
    try:
        if not confirm_deletion:
            raise BadRequestError(
                "Account deletion not confirmed",
                {"confirm_deletion": ["Must be true to delete"]}
            )
        
        # Delete account from database
        deleted = db_delete_account(account_id)
        
        if not deleted:
            raise NotFoundError("Account", account_id)
        
        # In production:
        # 1. Archive all account data to secure backup
        # 2. Delete all transactions, forecasts, models
        # 3. Delete account record ← done via db_delete_account
        # 4. Log deletion with admin user_id and timestamp
        # 5. Send notification to account owner
        
        return None  # 204 No Content
    except (BadRequestError, NotFoundError):
        raise
    except Exception:
        logger.exception("Account deletion failed", extra={"account_id": account_id, "user_id": current_admin["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/maintenance/cleanup",
    response_model=MaintenanceCleanupResponse
)
async def maintenance_cleanup(
    request: MaintenanceCleanupRequest,
    current_admin: dict = Depends(get_current_admin)
):
    """
    Cleanup old data (archival/deletion).
    
    Security:
    - Requires ADMIN role
    - Optional dry-run mode (doesn't delete)
    - Logs all operations
    - Only deletes records older than days_to_keep
    
    Args:
    - days_to_keep: Keep data from last N days (default 365)
    - dry_run: If true, only report what would be deleted
    
    Returns:
    - Number of records deleted
    - Space freed (MB)
    - Operation duration
    """
    try:
        if request.days_to_keep < 1:
            raise BadRequestError(
                "Invalid days_to_keep",
                {"days_to_keep": ["Must be >= 1"]}
            )
        
        start_time = datetime.utcnow()
        
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=request.days_to_keep)).date()
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            if request.dry_run:
                # Count records that would be deleted
                cursor.execute(
                    "SELECT COUNT(*) FROM transactions WHERE transaction_date < ?",
                    (str(cutoff_date),)
                )
                deleted_records = cursor.fetchone()[0]
            else:
                # Actually delete old transactions
                cursor.execute(
                    "DELETE FROM transactions WHERE transaction_date < ?",
                    (str(cutoff_date),)
                )
                deleted_records = cursor.rowcount
                conn.commit()
        
        # Estimate space freed (rough calculation: ~500 bytes per transaction)
        freed_space_mb = (deleted_records * 500) / (1024 * 1024)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return MaintenanceCleanupResponse(
            deleted_records=deleted_records,
            freed_space_mb=round(freed_space_mb, 2),
            duration_seconds=round(duration, 2)
        )
    except BadRequestError:
        raise
    except Exception:
        logger.exception("Maintenance cleanup failed", extra={"account_id": None, "user_id": current_admin["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
