"""Transaction management routes."""
from fastapi import APIRouter, Depends, status, File, UploadFile, Query, HTTPException
from datetime import datetime
from typing import List, Optional
import csv
from io import StringIO
import sqlite3
from pathlib import Path
import uuid
from src.api.schemas import (
    TransactionCreateRequest, TransactionBulkRequest, TransactionUpdateRequest,
    TransactionResponse, TransactionListResponse, BulkImportResponse
)
from src.api.auth import get_current_user
from src.api.exceptions import (
    NotFoundError, ForbiddenError, DuplicateError, ValidationError, FileUploadError
)
from src.db.repositories import verify_account_ownership, fetch_all_transactions
from src.monitoring.logger import logger

router = APIRouter()

DB_PATH = Path("data/finopsia.db")


def _resolve_category(description: str, amount: int, direction: str, category: Optional[str]) -> str:
    """Use provided category or auto-categorize with the API rule-based classifier."""
    if category:
        return str(category)

    from src.ml.categorizer import rule_based_category

    resolved = rule_based_category(description, direction)
    return resolved or "Miscellaneous"


def _verify_account_ownership(account_id: str, user_id: int):
    """Security: Verify user owns this account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")


@router.get("/{account_id}/transactions", response_model=TransactionListResponse)
async def list_transactions(
    account_id: str,
    current_user: dict = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    direction: Optional[str] = None,
):
    """
    List transactions for account with filters.
    
    Security:
    - Requires authentication
    - User can only access their own account's transactions
    - Pagination limits response size
    
    Query Parameters:
    - limit: Max 100 results per page
    - offset: Pagination offset
    - start_date, end_date: Date range filter
    - category: Filter by category
    - direction: Filter by inflow/outflow
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Get all transactions for this account
        all_txns = fetch_all_transactions(account_id, start_date, end_date)
        
        # Apply filters
        if category:
            all_txns = [tx for tx in all_txns if tx.get("category") == category]
        
        if direction:
            all_txns = [tx for tx in all_txns if tx["direction"] == direction]
        
        # Paginate
        total = len(all_txns)
        paginated = all_txns[offset:offset + limit]
        
        transactions = [TransactionResponse(**tx) for tx in paginated]
        
        return TransactionListResponse(
            transactions=transactions,
            total=total,
            limit=limit,
            offset=offset
        )
    except ForbiddenError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{account_id}/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    account_id: str,
    transaction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get single transaction details.
    
    Security:
    - Requires authentication
    - User can only access their own account's transactions
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT transaction_id, account_id, description, category, amount, direction, transaction_date, posted_at
                FROM transactions
                WHERE account_id = ? AND transaction_id = ?
                """,
                (account_id, transaction_id)
            )
            row = cursor.fetchone()
        
        if not row:
            raise NotFoundError("Transaction", transaction_id)
        
        tx = {
            "transaction_id": row[0],
            "account_id": row[1],
            "description": row[2],
            "category": row[3],
            "amount": row[4],
            "direction": row[5],
            "transaction_date": row[6],
            "posted_at": row[7],
        }
        
        return TransactionResponse(**tx)
    except ForbiddenError:
        raise
    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{account_id}/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(
    account_id: str,
    request: TransactionCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create single transaction.
    
    Security:
    - Requires authentication
    - User can only create in their own account
    - Validates all input fields
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Use provided ID or generate one
        transaction_id = request.transaction_id or str(uuid.uuid4())[:12]
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check for duplicates
            cursor.execute(
                "SELECT 1 FROM transactions WHERE account_id = ? AND transaction_id = ?",
                (account_id, transaction_id)
            )
            if cursor.fetchone():
                raise DuplicateError(f"Transaction {transaction_id} already exists")
            
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
                    request.description,
                    _resolve_category(request.description, request.amount, str(request.direction), request.category),
                    request.amount,
                    request.direction,
                    request.transaction_date,
                    request.posted_at or datetime.utcnow().isoformat() + "Z"
                )
            )
            conn.commit()
        
        return TransactionResponse(
            transaction_id=transaction_id,
            account_id=account_id,
            description=request.description,
            category=_resolve_category(request.description, request.amount, str(request.direction), request.category),
            amount=request.amount,
            direction=request.direction,
            transaction_date=request.transaction_date,
            posted_at=request.posted_at or datetime.utcnow().isoformat() + "Z"
        )
    except ForbiddenError:
        raise
    except DuplicateError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{account_id}/transactions/bulk", response_model=BulkImportResponse, status_code=status.HTTP_201_CREATED)
async def bulk_create_transactions(
    account_id: str,
    request: TransactionBulkRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Bulk create transactions (up to 1000).
    
    Security:
    - Requires authentication
    - User can only import to their own account
    - Validates all transactions
    - Skips duplicates
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        inserted = 0
        duplicates = 0
        errors = []
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            for idx, req in enumerate(request.transactions):
                try:
                    transaction_id = req.transaction_id or f"tx_bulk_{uuid.uuid4().hex[:8]}"
                    
                    # Check for duplicates
                    cursor.execute(
                        "SELECT 1 FROM transactions WHERE account_id = ? AND transaction_id = ?",
                        (account_id, transaction_id)
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
                            req.description,
                            _resolve_category(req.description, req.amount, str(req.direction), req.category),
                            req.amount,
                            req.direction,
                            req.transaction_date,
                            req.posted_at or datetime.utcnow().isoformat() + "Z"
                        )
                    )
                    inserted += 1
                    
                except Exception:
                    errors.append({
                        "index": idx,
                        "error": "Invalid transaction payload"
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
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{account_id}/transactions/upload-csv", response_model=BulkImportResponse, status_code=status.HTTP_201_CREATED)
async def upload_csv_transactions(
    account_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload transactions from CSV file.
    
    Security:
    - Requires authentication
    - File size limited
    - Only CSV format allowed
    - User can only upload to their own account
    
    CSV Format:
    transaction_id,description,category,amount,direction,transaction_date,posted_at
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
                    transaction_id = row.get("transaction_id") or f"tx_csv_{idx}"
                    
                    # Check for duplicates
                    cursor.execute(
                        "SELECT 1 FROM transactions WHERE account_id = ? AND transaction_id = ?",
                        (account_id, transaction_id)
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
                            row.get("transaction_date", datetime.now().isoformat()[:10]),
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
        logger.exception("CSV transaction upload failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{account_id}/transactions/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    account_id: str,
    transaction_id: str,
    request: TransactionUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update transaction (fix category, description, etc).
    
    Security:
    - Requires authentication
    - User can only update their own account's transactions
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Get existing transaction
            cursor.execute(
                """
                SELECT transaction_id, account_id, description, category, amount, direction, transaction_date, posted_at
                FROM transactions
                WHERE account_id = ? AND transaction_id = ?
                """,
                (account_id, transaction_id)
            )
            row = cursor.fetchone()
            
            if not row:
                raise NotFoundError("Transaction", transaction_id)
            
            # Prepare update values
            description = request.description if request.description else row[2]
            category = request.category if request.category else row[3]
            
            # Update in database
            cursor.execute(
                """
                UPDATE transactions
                SET description = ?, category = ?
                WHERE account_id = ? AND transaction_id = ?
                """,
                (description, category, account_id, transaction_id)
            )
            conn.commit()
        
        return TransactionResponse(
            transaction_id=row[0],
            account_id=row[1],
            description=description,
            category=category,
            amount=row[4],
            direction=row[5],
            transaction_date=row[6],
            posted_at=row[7],
        )
    except ForbiddenError:
        raise
    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{account_id}/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(
    account_id: str,
    transaction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete transaction.
    
    Security:
    - Requires authentication
    - User can only delete from their own account
    - Requires audit log entry (in production)
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM transactions WHERE account_id = ? AND transaction_id = ?",
                (account_id, transaction_id)
            )
            
            if cursor.rowcount == 0:
                raise NotFoundError("Transaction", transaction_id)
            
            conn.commit()
        
        return None  # 204 No Content
    except ForbiddenError:
        raise
    except NotFoundError:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
