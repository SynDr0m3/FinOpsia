"""FastAPI application initialization and configuration with security focus."""
import os

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from src.core.env import load_project_env
from src.api.rate_limit import limiter
from src.api.routes import (
    auth, accounts, transactions, categorization, forecasts,
    models, analytics, import_export, admin, health
)
from src.api.exceptions import APIException

load_project_env()


def create_app() -> FastAPI:
    """Create and configure FastAPI application with security."""
    app = FastAPI(
        title="FinOpsia API",
        description="Financial Operations Platform API - Secure",
        version="1.0.0",
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/_schema",
    )

    # ========================================================================
    # SECURITY: CORS Configuration - Restricted Origins
    # ========================================================================
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000"
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        max_age=3600,  # Cache preflight for 1 hour
    )

    # HTTPS redirect in production
    if os.getenv("ENVIRONMENT", "development") == "production":
        app.add_middleware(HTTPSRedirectMiddleware)

    # ========================================================================
    # SECURITY: Max Request Body Size (1MB limit)
    # ========================================================================
    @app.middleware("http")
    async def limit_upload_size(request, call_next):
        if request.method == "POST" or request.method == "PUT":
            max_body_size = 1_000_000  # 1MB
            if "content-length" in request.headers:
                content_length = int(request.headers["content-length"])
                if content_length > max_body_size:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"}
                    )
        return await call_next(request)

    # ========================================================================
    # SECURITY: Security Headers (XSS, Clickjacking, MIME type)
    # ========================================================================
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path.startswith("/api/v1/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "img-src 'self' data:; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self'"
            )
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

    @app.middleware("http")
    async def restrict_schema_access(request, call_next):
        if request.url.path == "/api/v1/_schema":
            referer = request.headers.get("referer", "")
            allowed_referers = (
                "http://127.0.0.1:8000/api/v1/docs",
                "http://localhost:8000/api/v1/docs",
            )
            if not any(referer.startswith(value) for value in allowed_referers):
                return JSONResponse(
                    status_code=404,
                    content={"detail": "Not found"},
                )
        return await call_next(request)

    # ========================================================================
    # SECURITY: Rate Limiting
    # ========================================================================
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please try again later.",
                "details": {},
            },
        )

    # ========================================================================
    # ERROR HANDLING - Standardized Error Responses
    # ========================================================================
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error,
                "message": exc.message,
                "details": exc.details,
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        errors = {}
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"][1:])
            errors[field] = error["msg"]
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors,
            }
        )

    # ========================================================================
    # ROUTES
    # ========================================================================

    # Health & Monitoring (no auth required)
    app.include_router(
        health.router,
        prefix="/api/v1/health",
        tags=["Health & Monitoring"]
    )

    # Authentication (no auth required)
    app.include_router(
        auth.router,
        prefix="/api/v1/users",
        tags=["Authentication & User Management"]
    )

    # Accounts (auth required)
    app.include_router(
        accounts.router,
        prefix="/api/v1/accounts",
        tags=["Account Management"],
        dependencies=[]
    )

    # Transactions (auth required)
    app.include_router(
        transactions.router,
        prefix="/api/v1",
        tags=["Transaction Management"]
    )

    # Categorization (auth required)
    app.include_router(
        categorization.router,
        prefix="/api/v1",
        tags=["Transaction Categorization"]
    )

    # Forecasting (auth required)
    app.include_router(
        forecasts.router,
        prefix="/api/v1",
        tags=["Balance Forecasting"]
    )

    # Model Management (auth required)
    app.include_router(
        models.router,
        prefix="/api/v1/models",
        tags=["Model Management"]
    )

    # Analytics (auth required)
    app.include_router(
        analytics.router,
        prefix="/api/v1",
        tags=["Analytics & Reporting"]
    )

    # Import/Export (auth required)
    app.include_router(
        import_export.router,
        prefix="/api/v1",
        tags=["Data Import/Export"]
    )

    # Admin Operations (admin role required)
    app.include_router(
        admin.router,
        prefix="/api/v1/admin",
        tags=["Admin Operations"]
    )

    return app


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )

