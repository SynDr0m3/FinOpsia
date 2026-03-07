"""Health check and monitoring endpoints."""
import json
import time
from fastapi import APIRouter, status, HTTPException, Depends
from datetime import datetime
from pathlib import Path
import psutil
from src.api.auth import get_current_admin
from src.automation.scheduler import scheduler
from src.db.repositories import _get_connection
from src.ml.persistence import MODEL_DIR
from src.api.schemas import HealthResponse, DetailedHealthResponse, LogsResponse, LogEntry

router = APIRouter()
LOG_PATH = Path("logs/finopsia.log")


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
    - 200: System is healthy
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed system health check.
    
    Checks:
    - Database connectivity
    - ML models availability
    - Scheduler status
    - System resources (CPU, memory, disk)
    
    Security:
    - Does not expose sensitive system info
    - Returns aggregated metrics only
    """
    try:
        db_start = time.perf_counter()
        with _get_connection() as conn:
            conn.execute("SELECT 1")
        db_latency_ms = int((time.perf_counter() - db_start) * 1000)

        categorizer_ready = (MODEL_DIR / "categorizer.joblib").exists()
        forecaster_count = len(list(MODEL_DIR.glob("forecaster_*.joblib")))
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return DetailedHealthResponse(
            status="healthy" if categorizer_ready else "degraded",
            services={
                "database": {
                    "status": "ok",
                    "latency_ms": db_latency_ms
                },
                "models": {
                    "status": "ok" if categorizer_ready else "degraded",
                    "ready": categorizer_ready,
                    "forecasters": forecaster_count,
                },
                "scheduler": {
                    "status": "running" if scheduler.running else "stopped",
                    "active_jobs": len(scheduler.get_jobs()),
                }
            },
            resources={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            }
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )


@router.get("/live")
async def liveness():
    """Liveness probe for Kubernetes."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get("/ready")
async def readiness():
    """Readiness probe for Kubernetes."""
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    limit: int = 100,
    level: str = None,
    account_id: str = None,
    user_id: int = None,
    start_date: str = None,
    end_date: str = None,
    current_admin: dict = Depends(get_current_admin),
):
    """
    Retrieve system logs with filtering.
    
    Query the real application log file with filters.
    """
    if limit > 1000:
        limit = 1000

    if not LOG_PATH.exists():
        return LogsResponse(logs=[], total=0, limit=limit)

    start_bound = datetime.fromisoformat(start_date).date() if start_date else None
    end_bound = datetime.fromisoformat(end_date).date() if end_date else None
    matched_logs: list[LogEntry] = []

    with LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            record = payload.get("record", {})
            record_level = record.get("level", {}).get("name")
            if level and record_level != level.upper():
                continue

            record_time = record.get("time", {}).get("repr")
            if record_time:
                record_dt = datetime.fromisoformat(record_time)
                if start_bound and record_dt.date() < start_bound:
                    continue
                if end_bound and record_dt.date() > end_bound:
                    continue
            else:
                record_dt = None

            extra = record.get("extra", {}).get("extra", {})
            record_account_id = extra.get("account_id")
            record_user_id = extra.get("user_id")
            if account_id and str(record_account_id) != str(account_id):
                continue
            if user_id is not None and record_user_id != user_id:
                continue

            matched_logs.append(
                LogEntry(
                    timestamp=record_dt.isoformat() + "Z" if record_dt else datetime.utcnow().isoformat() + "Z",
                    level=record_level or "INFO",
                    message=record.get("message", payload.get("text", "").strip()),
                    account_id=str(record_account_id) if record_account_id is not None else None,
                    user_id=record_user_id,
                )
            )

    matched_logs = list(reversed(matched_logs[-limit:]))
    return LogsResponse(logs=matched_logs, total=len(matched_logs), limit=limit)
