"""Transaction categorization routes."""
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, status, HTTPException
from typing import Optional
from src.api.schemas import (
    CategorizeRequest, CategorizeResponse, CategorizeBatchRequest, 
    CategorizeBatchResponse, CategorizeResult, AutoCategorizeResponse
)
from src.api.auth import get_current_user
from src.api.exceptions import NotFoundError, ForbiddenError, ValidationError
from src.db.repositories import verify_account_ownership

router = APIRouter()
DB_PATH = Path("data/finopsia.db")


def _verify_account_ownership(account_id: str, user_id: int) -> None:
    """Security: verify the authenticated user owns the account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")

# Mock categorization service
def _categorize(description: str, amount: int, direction: str) -> tuple[str, float]:
    """
    Categorize transaction using rule-based system.
    
    Returns: (category, confidence)
    """
    from src.ml.categorizer import rule_based_category
    
    category = rule_based_category(description, direction)
    if not category:
        category = "Miscellaneous"
    
    # Simple confidence scoring
    confidence = 0.85 if category != "Miscellaneous" else 0.6
    
    return category, confidence


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_single(
    request: CategorizeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Categorize a single transaction.
    
    Security:
    - Requires authentication
    - No sensitive data exposure
    
    Returns category and confidence score.
    """
    category, confidence = _categorize(
        request.description,
        request.amount,
        request.direction
    )
    
    return CategorizeResponse(
        category=category,
        confidence=round(confidence, 2),
        method="rule_based"
    )


@router.post("/categorize/batch", response_model=CategorizeBatchResponse)
async def categorize_batch(
    request: CategorizeBatchRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Categorize multiple transactions (up to 1000).
    
    Security:
    - Requires authentication
    - Processes up to 1000 transactions
    - Returns consistent results
    """
    results = []
    
    for txn in request.transactions:
        category, confidence = _categorize(
            txn.description,
            txn.amount,
            txn.direction
        )
        
        results.append(CategorizeResult(
            description=txn.description,
            category=category,
            confidence=round(confidence, 2)
        ))
    
    return CategorizeBatchResponse(results=results)


@router.post("/accounts/{account_id}/transactions/categorize", response_model=AutoCategorizeResponse)
async def auto_categorize_account(
    account_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Auto-categorize all uncategorized transactions for an account.
    
    Security:
    - Requires authentication
    - User can only categorize their own account's transactions
    
    Returns count of categorized transactions.
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])

        categorized_count = 0
        already_categorized_count = 0

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT transaction_id, description, amount, direction, category
                FROM transactions
                WHERE account_id = ?
                """,
                (account_id,),
            )
            rows = cursor.fetchall()

            for transaction_id, description, amount, direction, category in rows:
                if category:
                    already_categorized_count += 1
                    continue

                resolved_category, _ = _categorize(description, amount, direction)
                cursor.execute(
                    """
                    UPDATE transactions
                    SET category = ?
                    WHERE account_id = ? AND transaction_id = ?
                    """,
                    (resolved_category, account_id, transaction_id),
                )
                categorized_count += 1

            conn.commit()

        return AutoCategorizeResponse(
            categorized=categorized_count,
            already_categorized=already_categorized_count,
            message=f"{categorized_count} transactions categorized",
        )
    except ForbiddenError:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
