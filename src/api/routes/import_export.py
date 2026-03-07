"""Data import/export routes for CSV and JSON."""
from fastapi import APIRouter, Depends, Query, File, UploadFile, status, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from datetime import datetime
from io import StringIO, BytesIO
import csv
import json
import sqlite3
from pathlib import Path
from src.api.schemas import BulkImportResponse, ErrorResponse
from src.api.auth import get_current_user
from src.api.exceptions import NotFoundError, ForbiddenError, FileUploadError
from src.db.repositories import verify_account_ownership, fetch_all_transactions
from src.monitoring.logger import logger

router = APIRouter()

DB_PATH = Path("data/finopsia.db")


def _resolve_category(description: str, amount: int, direction: str, category: Optional[str]) -> str:
    """Use provided category or auto-categorize from transaction details."""
    if category:
        return category

    from src.ml.categorizer import rule_based_category

    resolved = rule_based_category(description, direction)
    return resolved or "Miscellaneous"


def _verify_account_ownership(account_id: str, user_id: int):
    """Security: Verify user owns this account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")


@router.get("/{account_id}/export/csv")
async def export_csv(
    account_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Export transactions as CSV file.
    
    Security:
    - Requires authentication
    - User can only export their own account's data
    - File is streamed (not stored)
    
    Query Parameters:
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    
    Returns:
    - CSV file with transactions
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Get transactions for account
        txns = fetch_all_transactions(account_id, start_date, end_date)
        
        if len(txns) == 0:
            # Return empty CSV with headers
            output = StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "transaction_id", "account_id", "description", "category",
                    "amount", "direction", "transaction_date", "posted_at"
                ]
            )
            writer.writeheader()
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=transactions_{account_id}.csv"}
            )
        
        # Create CSV
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "transaction_id", "account_id", "description", "category",
                "amount", "direction", "transaction_date", "posted_at"
            ]
        )
        
        writer.writeheader()
        for tx in txns:
            writer.writerow(tx)
        
        # Return as file download
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=transactions_{account_id}.csv"}
        )
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("CSV export failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{account_id}/export/json")
async def export_json(
    account_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Export transactions as JSON.
    
    Security:
    - Requires authentication
    - User can only export their own account's data
    
    Query Parameters:
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    
    Returns:
    - JSON object with transactions array
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Get transactions for account
        txns = fetch_all_transactions(account_id, start_date, end_date)
        
        return {
            "account_id": account_id,
            "export_date": datetime.utcnow().isoformat() + "Z",
            "transaction_count": len(txns),
            "transactions": txns
        }
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("JSON export failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{account_id}/import/csv", response_model=BulkImportResponse)
async def import_csv(
    account_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Import transactions from CSV file.
    
    Security:
    - Requires authentication
    - User can only import to their own account
    - File size limited to 10MB
    - CSV format validated
    - Duplicates skipped
    
    CSV Format:
    transaction_id,description,category,amount,direction,transaction_date,posted_at
    
    Returns:
    - Count of inserted, skipped, error transactions
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Validate file
        if not file.filename.endswith(".csv"):
            raise FileUploadError("Only CSV files are allowed")
        
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        
        if size_mb > 10:
            raise FileUploadError("File size exceeds 10MB limit")
        
        # Parse CSV
        text = contents.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        
        inserted = 0
        duplicates = 0
        errors = []
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            for idx, row in enumerate(reader):
                try:
                    transaction_id = row.get("transaction_id")
                    if not transaction_id:
                        raise ValueError("Missing transaction_id")
                    
                    # Check for duplicates
                    cursor.execute(
                        "SELECT 1 FROM transactions WHERE transaction_id = ?",
                        (transaction_id,)
                    )
                    if cursor.fetchone():
                        duplicates += 1
                        continue
                    
                    # Insert transaction
                    cursor.execute(
                        """
                        INSERT INTO transactions 
                        (transaction_id, account_id, description, category, amount, direction, transaction_date, posted_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            transaction_id,
                            account_id,
                            row.get("description", ""),
                            _resolve_category(
                                row.get("description", ""),
                                int(row.get("amount", 0)),
                                row.get("direction", "outflow"),
                                row.get("category"),
                            ),
                            int(row.get("amount", 0)),
                            row.get("direction", "outflow"),
                            row.get("transaction_date", ""),
                            row.get("posted_at", datetime.utcnow().isoformat() + "Z")
                        )
                    )
                    inserted += 1
                    
                except Exception:
                    errors.append({
                        "row": idx + 2,
                        "error": "Invalid transaction row"
                    })
            
            conn.commit()
        
        return BulkImportResponse(
            inserted=inserted,
            duplicates_skipped=duplicates,
            errors=errors,
            message=f"Successfully imported {inserted} transactions"
        )
        
    except ForbiddenError:
        raise
    except FileUploadError:
        raise
    except Exception:
        logger.exception("CSV import failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
