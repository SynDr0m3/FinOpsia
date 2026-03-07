"""API exceptions and error handling with security focus."""
from fastapi import HTTPException, status
from typing import Optional, Any, Dict


class APIException(HTTPException):
    """Base API exception."""
    
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details or {}
        
        super().__init__(
            status_code=status_code,
            detail={
                "error": error,
                "message": message,
                "details": self.details,
            }
        )


class ValidationError(APIException):
    """Validation error (422)."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="VALIDATION_ERROR",
            message=message,
            details=details,
        )


class UnauthorizedError(APIException):
    """Unauthorized error (401)."""
    def __init__(self, message: str = "Invalid or expired token"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error="UNAUTHORIZED",
            message=message,
        )


class ForbiddenError(APIException):
    """Forbidden error (403)."""
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error="FORBIDDEN",
            message=message,
        )


class NotFoundError(APIException):
    """Not found error (404)."""
    def __init__(self, resource: str, resource_id: Any = None):
        if resource_id:
            msg = f"{resource} {resource_id} does not exist"
        else:
            msg = f"{resource} not found"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error="NOT_FOUND",
            message=msg,
        )


class DuplicateError(APIException):
    """Duplicate resource error (409)."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error="DUPLICATE",
            message=message,
            details=details,
        )


class BadRequestError(APIException):
    """Bad request error (400)."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="INVALID_REQUEST",
            message=message,
            details=details,
        )


class DatabaseError(APIException):
    """Database error (500)."""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="INTERNAL_ERROR",
            message="An unexpected error occurred. Please contact support.",
        )


class FileUploadError(APIException):
    """File upload error (400)."""
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="FILE_UPLOAD_ERROR",
            message=message,
        )


class RateLimitError(APIException):
    """Rate limit exceeded (429)."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            error="RATE_LIMIT_EXCEEDED",
            message=message,
        )


# Legacy compatibility
class FinOpsiaException(APIException):
    """Base exception for FinOpsia API (legacy)."""
    
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(
            status_code=status_code,
            error="ERROR",
            message=detail,
        )

class LegacyValidationError(FinOpsiaException):
    """Legacy validation error exception."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class InternalServerError(FinOpsiaException):
    """Internal server error exception."""
    
    def __init__(self, detail: str):
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
