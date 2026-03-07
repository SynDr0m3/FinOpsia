"""Pydantic schemas for request/response validation with security focus."""
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
#  ENUMS
# ============================================================================

class DirectionEnum(str, Enum):
    """Transaction direction."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class CategoryEnum(str, Enum):
    """Transaction categories."""
    SALARIES = "Salaries"
    RENT = "Rent"
    UTILITIES = "Utilities"
    REVENUE = "Revenue"
    INVENTORY = "Inventory"
    MARKETING = "Marketing"
    SUPPLIES = "Supplies"
    MISCELLANEOUS = "Miscellaneous"


class ModelStatusEnum(str, Enum):
    """Model training status."""
    TRAINED = "trained"
    MISSING = "missing"
    IN_TRAINING = "training"
    FAILED = "failed"


# ============================================================================
#  AUTH & USER SCHEMAS
# ============================================================================

class UserLoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserLoginResponse(BaseModel):
    """User login response."""
    access_token: str
    user_id: int
    username: str
    role: str


class UserResponse(BaseModel):
    """User profile response."""
    user_id: int
    username: str
    email: str
    role: str


class UserLogoutResponse(BaseModel):
    """User logout response."""
    message: str


# ============================================================================
#  ACCOUNT SCHEMAS
# ============================================================================

class AccountCreateRequest(BaseModel):
    """Create account request."""
    account_number: str = Field(..., min_length=5, max_length=20)
    starting_balance: int = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)

    @validator("account_number")
    def validate_account_number(cls, v):
        if not v.replace(" ", "").isalnum():
            raise ValueError("Account number must be alphanumeric")
        return v.strip()


class AccountUpdateRequest(BaseModel):
    """Update account request."""
    starting_balance: Optional[int] = Field(None, gt=0)


class AccountResponse(BaseModel):
    """Account response."""
    account_id: str
    account_number: str
    starting_balance: int
    currency: str
    last_updated: str


class AccountsListResponse(BaseModel):
    """List accounts response."""
    accounts: List[AccountResponse]


# ============================================================================
#  TRANSACTION SCHEMAS
# ============================================================================

class TransactionBase(BaseModel):
    """Base transaction schema."""
    description: str = Field(..., min_length=1, max_length=500)
    category: Optional[CategoryEnum] = None
    amount: int = Field(..., gt=0)
    direction: DirectionEnum
    transaction_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    posted_at: Optional[str] = None


class TransactionCreateRequest(TransactionBase):
    """Create transaction request."""
    transaction_id: Optional[str] = Field(None, max_length=100)


class TransactionBulkRequest(BaseModel):
    """Bulk transaction create request."""
    transactions: List[TransactionCreateRequest] = Field(..., min_items=1, max_items=1000)


class TransactionUpdateRequest(BaseModel):
    """Update transaction request."""
    category: Optional[CategoryEnum] = None
    description: Optional[str] = Field(None, min_length=1, max_length=500)


class TransactionResponse(BaseModel):
    """Transaction response."""
    transaction_id: str
    account_id: str
    description: str
    category: Optional[str]
    amount: int
    direction: str
    transaction_date: str
    posted_at: str


class TransactionListResponse(BaseModel):
    """List transactions response with pagination."""
    transactions: List[TransactionResponse]
    total: int
    limit: int
    offset: int


class BulkImportResponse(BaseModel):
    """Bulk import response."""
    inserted: int
    duplicates_skipped: int
    errors: List[Dict[str, Any]] = []
    message: str


# ============================================================================
#  CATEGORIZATION SCHEMAS
# ============================================================================

class CategorizeRequest(BaseModel):
    """Categorize single transaction request."""
    description: str = Field(..., min_length=1, max_length=500)
    amount: int = Field(..., gt=0)
    direction: DirectionEnum


class CategorizeResponse(BaseModel):
    """Categorize response."""
    category: str
    confidence: float = Field(..., ge=0, le=1)
    method: str


class CategorizeBatchRequest(BaseModel):
    """Categorize batch request."""
    transactions: List[CategorizeRequest] = Field(..., min_items=1, max_items=1000)


class CategorizeResult(BaseModel):
    """Single categorization result."""
    description: str
    category: str
    confidence: float


class CategorizeBatchResponse(BaseModel):
    """Categorize batch response."""
    results: List[CategorizeResult]


class AutoCategorizeResponse(BaseModel):
    """Auto categorize response."""
    categorized: int
    already_categorized: int
    message: str


# ============================================================================
#  FORECASTING SCHEMAS
# ============================================================================

class ForecastPoint(BaseModel):
    """Single forecast point."""
    date: str
    balance: int
    lower_bound: int
    upper_bound: int
    confidence_interval: float


class ForecastResponse(BaseModel):
    """Balance forecast response."""
    account_id: str
    forecast_date: str
    days_ahead: int
    forecasts: List[ForecastPoint]
    model_status: str
    trained_at: Optional[str] = None


class ForecastGenerateRequest(BaseModel):
    """Generate forecast request."""
    days_ahead: int = Field(default=7, ge=1, le=90)


class ForecastGenerateResponse(BaseModel):
    """Generate forecast response."""
    message: str
    days_ahead: int
    forecast_count: int


# ============================================================================
#  MODEL MANAGEMENT SCHEMAS
# ============================================================================

class CategorizerStatus(BaseModel):
    """Categorizer model status."""
    status: str
    trained_at: Optional[str] = None
    version: str
    accuracy: Optional[float] = None


class ForecasterStatus(BaseModel):
    """Forecaster model status."""
    account_id: str
    status: str
    trained_at: Optional[str] = None
    days_history: Optional[int] = None
    mape: Optional[float] = None
    last_error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    """Model status response."""
    categorizer: CategorizerStatus
    forecasters: List[ForecasterStatus]


class CategorizerDetailsResponse(BaseModel):
    """Categorizer model details."""
    model_type: str
    status: str
    trained_at: Optional[str] = None
    version: str
    categories: List[str]
    performance: Dict[str, float]
    training_samples: int


class RetrainingTrainingSample(BaseModel):
    """Training sample for retraining."""
    description: str = Field(..., min_length=1, max_length=500)
    category: CategoryEnum


class CategorizerRetrainRequest(BaseModel):
    """Retrain categorizer request."""
    training_samples: List[RetrainingTrainingSample] = Field(..., min_items=1, max_items=5000)


class RetrainingJobResponse(BaseModel):
    """Retraining job response."""
    job_id: str
    status: str
    message: str
    account_id: Optional[str] = None
    trained_at: Optional[str] = None
    performance: Optional[Dict[str, float]] = None


# ============================================================================
#  ANALYTICS SCHEMAS
# ============================================================================

class CategoryBreakdown(BaseModel):
    """Category breakdown."""
    category: str
    count: int
    total_amount: int
    percentage: float
    avg_amount: int
    trend: str


class PeriodData(BaseModel):
    """Period data for category."""
    period: str
    amount: int
    count: int


class CategoryAnalytics(BaseModel):
    """Category analytics."""
    category: str
    count: int
    total_amount: int
    percentage: float
    avg_amount: int
    trend: str
    by_period: List[PeriodData]


class AccountSummaryResponse(BaseModel):
    """Account summary response."""
    account_id: str
    period: Dict[str, str]
    total_inflow: int
    total_outflow: int
    net_change: int
    current_balance: int
    avg_daily_balance: int
    transaction_count: int
    largest_inflow: Dict[str, Any]
    largest_outflow: Dict[str, Any]
    categories_breakdown: Dict[str, Dict[str, Any]]


class CategoriesAnalyticsResponse(BaseModel):
    """Category analytics response."""
    account_id: str
    period: Dict[str, str]
    categories: List[CategoryAnalytics]


class DailyFlow(BaseModel):
    """Daily cash flow."""
    date: str
    inflow: int
    outflow: int
    net: int


class CashflowAnalyticsResponse(BaseModel):
    """Cashflow analysis response."""
    account_id: str
    daily_cashflow: List[DailyFlow]
    weekly_summary: List[Dict[str, Any]]
    volatility: float
    trend: str


# ============================================================================
#  HEALTH & MONITORING SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    """Basic health check response."""
    status: str
    timestamp: str


class ServiceStatus(BaseModel):
    """Service status."""
    status: str
    latency_ms: Optional[int] = None
    ready: Optional[bool] = None
    active_jobs: Optional[int] = None


class ResourceInfo(BaseModel):
    """Resource information."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float


class DetailedHealthResponse(BaseModel):
    """Detailed health response."""
    status: str
    services: Dict[str, Dict[str, Any]]
    resources: ResourceInfo


class LogEntry(BaseModel):
    """Log entry."""
    timestamp: str
    level: str
    message: str
    account_id: Optional[str] = None
    user_id: Optional[int] = None


class LogsResponse(BaseModel):
    """Logs response."""
    logs: List[LogEntry]
    total: int
    limit: int


# ============================================================================
#  ADMIN SCHEMAS
# ============================================================================

class ResetPasswordRequest(BaseModel):
    """Admin password reset request."""
    new_password: str = Field(..., min_length=12, max_length=128)


class ResetPasswordResponse(BaseModel):
    """Reset password response."""
    message: str


class DeleteAccountRequest(BaseModel):
    """Delete account request (requires confirmation)."""
    confirm_deletion: bool


class MaintenanceCleanupRequest(BaseModel):
    """Maintenance cleanup request."""
    days_to_keep: int = Field(default=365, ge=1)
    dry_run: bool = False


class MaintenanceCleanupResponse(BaseModel):
    """Maintenance cleanup response."""
    deleted_records: int
    freed_space_mb: float
    duration_seconds: float


# ============================================================================
#  ERROR RESPONSE SCHEMAS
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail."""
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ============================================================================
#  PAGINATION HELPER
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters."""
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
