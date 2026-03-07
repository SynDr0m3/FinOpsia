"""Account management routes."""
from fastapi import APIRouter, Depends, status, HTTPException, Query
from datetime import datetime
import uuid
from src.api.schemas import (
    AccountCreateRequest, AccountUpdateRequest, AccountResponse, 
    AccountsListResponse
)
from src.api.auth import get_current_user
from src.api.exceptions import NotFoundError, ForbiddenError, DuplicateError, ValidationError
from src.api.dependencies import verify_user_account_access
from src.db.repositories import (
    fetch_user_accounts, fetch_account_metadata, verify_account_ownership,
    create_account as db_create_account, update_account as db_update_account,
    delete_account
)
from src.monitoring.logger import logger

router = APIRouter()


@router.get("/", response_model=AccountsListResponse)
async def list_accounts(current_user: dict = Depends(get_current_user)):
    """
    List all accounts for current user.
    
    Security:
    - Requires authentication
    - Returns only user's own accounts
    """
    try:
        user_accounts = fetch_user_accounts(current_user["user_id"])
        return AccountsListResponse(accounts=[AccountResponse(**acc) for acc in user_accounts])
    except Exception:
        logger.exception("List accounts failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{account_id}", response_model=AccountResponse)
async def get_account(
    account_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get account metadata.
    
    Security:
    - Requires authentication
    - User can only access their own accounts
    
    Raises:
    - 404: Account not found
    - 403: User does not own account
    """
    try:
        # Verify ownership first
        if not verify_account_ownership(account_id, current_user["user_id"]):
            raise ForbiddenError(f"User does not have access to account_id: {account_id}")
        
        account = fetch_account_metadata(account_id, current_user["user_id"])
        return AccountResponse(**account)
    except ValueError:
        raise NotFoundError("Account", account_id)
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("Get account failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
async def create_account(
    request: AccountCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create new account.
    
    Security:
    - Requires authentication
    - Validates all input fields
    - Checks for duplicate account numbers
    
    Raises:
    - 409: Account number already exists
    - 422: Validation error
    """
    # Validate currency code (ISO 4217)
    if not request.currency.isupper() or len(request.currency) != 3:
        raise ValidationError(
            "Invalid currency code",
            {"currency": ["Must be 3-letter ISO code (e.g., NGN, USD)"]}
        )
    
    try:
        # Create account with auto-generated ID
        new_account_id = str(uuid.uuid4())[:8]  # Use shorter UUID
        
        account = db_create_account(
            account_id=new_account_id,
            user_id=current_user["user_id"],
            account_number=request.account_number,
            starting_balance=request.starting_balance,
            currency=request.currency
        )
        
        return AccountResponse(**account)
    except ValueError as exc:
        logger.warning("Account creation validation failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        if "already exists" in str(exc):
            raise DuplicateError("Account already exists")
        raise ValidationError("Account creation failed", {"account": ["Invalid account data"]})
    except Exception:
        logger.exception("Create account failed", extra={"account_id": None, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{account_id}", response_model=AccountResponse)
async def update_account(
    account_id: str,
    request: AccountUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update account.
    
    Security:
    - Requires authentication
    - User can only update their own accounts
    
    Raises:
    - 404: Account not found
    - 403: User does not own account
    - 422: Validation error
    """
    try:
        # Verify ownership first
        if not verify_account_ownership(account_id, current_user["user_id"]):
            raise ForbiddenError(f"User does not have access to account_id: {account_id}")
        
        # Build update dict
        update_dict = {}
        
        if request.starting_balance is not None:
            if request.starting_balance <= 0:
                raise ValidationError(
                    "Invalid starting balance",
                    {"starting_balance": ["Must be greater than 0"]}
                )
            update_dict["starting_balance"] = request.starting_balance
        
        if request.currency is not None:
            if not request.currency.isupper() or len(request.currency) != 3:
                raise ValidationError(
                    "Invalid currency code",
                    {"currency": ["Must be 3-letter ISO code (e.g., NGN, USD)"]}
                )
            update_dict["currency"] = request.currency
        
        account = db_update_account(account_id, **update_dict)

        return AccountResponse(**account)
    except ValueError:
        raise NotFoundError("Account", account_id)
    except (NotFoundError, ForbiddenError, ValidationError):
        raise
    except Exception:
        logger.exception("Update account failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
