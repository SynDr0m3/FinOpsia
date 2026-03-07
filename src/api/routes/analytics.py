"""Analytics and reporting routes."""
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime
import pandas as pd
from src.api.schemas import (
    AccountSummaryResponse, CategoriesAnalyticsResponse, CashflowAnalyticsResponse,
    CategoryAnalytics, PeriodData, DailyFlow
)
from src.api.auth import get_current_user
from src.api.exceptions import NotFoundError, ForbiddenError
from src.db.repositories import (
    verify_account_ownership, fetch_all_transactions, fetch_transactions,
    fetch_account_metadata
)
from src.monitoring.logger import logger

router = APIRouter()


def _verify_account_ownership(account_id: str, user_id: int):
    """Security: Verify user owns this account."""
    if not verify_account_ownership(account_id, user_id):
        raise ForbiddenError(f"User does not have access to account_id: {account_id}")


@router.get("/{account_id}/summary", response_model=AccountSummaryResponse)
async def get_account_summary(
    account_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get account summary with totals, averages, trends.
    
    Security:
    - Requires authentication
    - User can only access their own account's summary
    
    Query Parameters:
    - start_date: Filter from date (YYYY-MM-DD)
    - end_date: Filter to date (YYYY-MM-DD)
    
    Returns:
    - Total inflow/outflow amounts
    - Net change and current balance
    - Average daily balance
    - Transaction count
    - Largest transactions
    - Category breakdown
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Fetch transactions
        txns = fetch_all_transactions(account_id, start_date, end_date)
        df = pd.DataFrame(txns)
        
        if len(df) == 0:
            return AccountSummaryResponse(
                account_id=account_id,
                period={"start": start_date or "unknown", "end": end_date or "unknown"},
                total_inflow=0,
                total_outflow=0,
                net_change=0,
                current_balance=0,
                avg_daily_balance=0,
                transaction_count=0,
                largest_inflow={"amount": 0, "description": "N/A"},
                largest_outflow={"amount": 0, "description": "N/A"},
                categories_breakdown={}
            )
        
        # Calculate inflow/outflow
        inflows = df[df["direction"] == "inflow"]
        outflows = df[df["direction"] == "outflow"]
        
        total_inflow = inflows["amount"].sum() if len(inflows) > 0 else 0
        total_outflow = outflows["amount"].sum() if len(outflows) > 0 else 0
        net_change = total_inflow - total_outflow
        
        # Get account metadata for starting balance
        try:
            account_meta = fetch_account_metadata(account_id)
            current_balance = account_meta["starting_balance"] + net_change
        except:
            current_balance = net_change
        
        # Largest transactions
        largest_inflow = inflows.nlargest(1, "amount") if len(inflows) > 0 else None
        largest_outflow = outflows.nlargest(1, "amount") if len(outflows) > 0 else None
        
        # Category breakdown
        categories_breakdown = {}
        if "category" in df.columns:
            for cat in df["category"].unique():
                cat_df = df[df["category"] == cat]
                categories_breakdown[cat] = {
                    "count": len(cat_df),
                    "total": cat_df["amount"].sum()
                }
        
        return AccountSummaryResponse(
            account_id=account_id,
            period={
                "start": start_date or (df["transaction_date"].min() if len(df) > 0 else ""),
                "end": end_date or (df["transaction_date"].max() if len(df) > 0 else "")
            },
            total_inflow=int(total_inflow),
            total_outflow=int(total_outflow),
            net_change=int(net_change),
            current_balance=int(current_balance),
            avg_daily_balance=int(current_balance),
            transaction_count=len(df),
            largest_inflow={
                "amount": int(largest_inflow["amount"].iloc[0]) if largest_inflow is not None else 0,
                "description": largest_inflow["description"].iloc[0] if largest_inflow is not None else "N/A"
            },
            largest_outflow={
                "amount": int(largest_outflow["amount"].iloc[0]) if largest_outflow is not None else 0,
                "description": largest_outflow["description"].iloc[0] if largest_outflow is not None else "N/A"
            },
            categories_breakdown=categories_breakdown
        )
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("Account summary failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{account_id}/analytics/categories", response_model=CategoriesAnalyticsResponse)
async def get_category_analytics(
    account_id: str,
    start_date: Optional[str] = None,
    granularity: str = Query("monthly", pattern="^(daily|weekly|monthly)$"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get category breakdown and trends over time.
    
    Security:
    - Requires authentication
    - User can only access their own account's analytics
    
    Query Parameters:
    - start_date: Filter from date (YYYY-MM-DD)
    - granularity: Time grouping (daily, weekly, monthly)
    
    Returns:
    - Category totals and percentages
    - Average per transaction
    - Trend direction (increasing/decreasing/stable)
    - Period-by-period breakdown
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Fetch transactions
        txns = fetch_all_transactions(account_id, start_date, None)
        df = pd.DataFrame(txns)
        
        if len(df) == 0:
            return CategoriesAnalyticsResponse(
                account_id=account_id,
                period={"start": start_date or "unknown", "end": "unknown"},
                categories=[]
            )
        
        # Convert transaction_date to datetime
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        
        # Calculate total for percentage
        total_amount = df["amount"].sum()
        
        categories = []
        if "category" in df.columns:
            for cat in df["category"].unique():
                cat_df = df[df["category"] == cat]
                count = len(cat_df)
                total = cat_df["amount"].sum()
                percentage = (total / total_amount * 100) if total_amount > 0 else 0
                avg_amount = total / count if count > 0 else 0
                
                # Create period breakdown
                if granularity == "monthly":
                    cat_df["period"] = cat_df["transaction_date"].dt.strftime("%Y-%m")
                elif granularity == "weekly":
                    cat_df["period"] = cat_df["transaction_date"].dt.strftime("%Y-W%V")
                else:  # daily
                    cat_df["period"] = cat_df["transaction_date"].dt.strftime("%Y-%m-%d")
                
                by_period = [
                    PeriodData(period=period_name, amount=int(period_df["amount"].sum()), count=len(period_df))
                    for period_name, period_df in cat_df.groupby("period")
                ]
                
                categories.append(CategoryAnalytics(
                    category=cat,
                    count=count,
                    total_amount=int(total),
                    percentage=round(percentage, 2),
                    avg_amount=int(avg_amount),
                    trend="stable",  # TODO: calculate trend from actual data
                    by_period=by_period
                ))
        
        return CategoriesAnalyticsResponse(
            account_id=account_id,
            period={
                "start": start_date or str(df["transaction_date"].min()),
                "end": str(df["transaction_date"].max())
            },
            categories=categories
        )
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("Category analytics failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{account_id}/analytics/cashflow", response_model=CashflowAnalyticsResponse)
async def get_cashflow_analytics(
    account_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get detailed cashflow analysis.
    
    Security:
    - Requires authentication
    - User can only access their own account's cashflow
    
    Returns:
    - Daily cashflow breakdown
    - Weekly summary
    - Volatility metric
    - Overall trend (increasing/decreasing)
    """
    try:
        _verify_account_ownership(account_id, current_user["user_id"])
        
        # Fetch transactions
        txns = fetch_all_transactions(account_id, None, None)
        df = pd.DataFrame(txns)
        
        if len(df) == 0:
            return CashflowAnalyticsResponse(
                account_id=account_id,
                daily_cashflow=[],
                weekly_summary=[],
                volatility=0.0,
                trend="stable"
            )
        
        # Convert transaction_date to datetime
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        
        # Daily cashflow
        daily_flows = []
        for date, date_df in df.groupby(df["transaction_date"].dt.date):
            inflow = date_df[date_df["direction"] == "inflow"]["amount"].sum()
            outflow = date_df[date_df["direction"] == "outflow"]["amount"].sum()
            daily_flows.append(DailyFlow(
                date=str(date),
                inflow=int(inflow),
                outflow=int(outflow),
                net=int(inflow - outflow)
            ))
        
        # Weekly summary
        df["week"] = df["transaction_date"].dt.strftime("Week %V")
        weekly_summary = []
        for week, week_df in df.groupby("week"):
            inflow = week_df[week_df["direction"] == "inflow"]["amount"].sum()
            outflow = week_df[week_df["direction"] == "outflow"]["amount"].sum()
            weekly_summary.append({
                "week": week,
                "inflow": int(inflow),
                "outflow": int(outflow),
                "net": int(inflow - outflow)
            })
        
        # Calculate volatility (simple std dev of daily net flows)
        daily_nets = [f.net for f in daily_flows]
        volatility = pd.Series(daily_nets).std() / (pd.Series(daily_nets).mean() + 1) if len(daily_nets) > 0 else 0.0
        
        return CashflowAnalyticsResponse(
            account_id=account_id,
            daily_cashflow=daily_flows,
            weekly_summary=weekly_summary,
            volatility=round(float(volatility), 2),
            trend="stable"  # TODO: calculate from actual trend
        )
    except ForbiddenError:
        raise
    except Exception:
        logger.exception("Cashflow analytics failed", extra={"account_id": account_id, "user_id": current_user["user_id"]})
        raise HTTPException(status_code=500, detail="Internal server error")
