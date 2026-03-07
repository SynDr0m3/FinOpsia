"""Authentication and user management routes."""
from fastapi import APIRouter, HTTPException, status, Depends, Request
from datetime import timedelta
from src.api.schemas import (
    UserLoginRequest, UserLoginResponse, UserResponse, UserLogoutResponse
)
from src.api.auth import (
    create_access_token, hash_password, verify_password, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.api.exceptions import UnauthorizedError, ValidationError
from src.api.rate_limit import limiter
from src.db.repositories import fetch_user_by_username
from src.monitoring.logger import logger

router = APIRouter()


@router.post("/login", response_model=UserLoginResponse)
@limiter.limit("5/minute")
async def login(request: Request, credentials: UserLoginRequest):
    """
    User login endpoint.
    
    Security:
    - Validates username & password
    - Returns JWT access token
    - Tokens expire after 30 minutes
    """
    try:
        user = fetch_user_by_username(credentials.username)
    except Exception:
        logger.exception("Failed during login lookup", extra={"account_id": None, "user_id": None})
        user = None
    
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise UnauthorizedError("Invalid username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["user_id"]), "username": user["username"]},
        expires_delta=access_token_expires
    )
    
    return UserLoginResponse(
        access_token=access_token,
        user_id=user["user_id"],
        username=user["username"],
        role=user.get("role", "user"),
    )


@router.post("/logout", response_model=UserLogoutResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    User logout endpoint.
    
    Security:
    - Requires valid JWT token
    - In production, could implement token blacklisting
    """
    return UserLogoutResponse(message="Logged out successfully")


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile.
    
    Security:
    - Requires valid JWT token
    - Returns only current user's data
    """
    from src.db.repositories import fetch_user_by_id
    
    try:
        user = fetch_user_by_id(current_user["user_id"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            role=user.get("role", "user"),
        )
    except Exception:
        logger.exception("Failed to fetch current user profile", extra={"account_id": None, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
