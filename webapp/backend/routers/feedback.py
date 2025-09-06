#!/usr/bin/env python3
"""
Feedback router for handling user feedback endpoints.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Depends, Query

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import FeedbackRequest, FeedbackResponse

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["feedback"])

# In-memory feedback storage (fallback)
feedback_storage: Dict[str, Dict] = {}

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a report"""
    try:
        logger.info(f"📝 Submitting feedback for report: {request.report_id}")
        
        feedback_id = str(uuid.uuid4())
        
        # Store feedback
        feedback_data = {
            "feedback_id": feedback_id,
            "report_id": request.report_id,
            "feedback_type": request.feedback_type,
            "category": request.category,
            "rating": request.rating,
            "comment": request.comment,
            "user_session_id": request.user_session_id,
            "feedback_context": request.feedback_context,
            "created_at": datetime.now(),
            "langsmith_submitted": False
        }
        
        feedback_storage[feedback_id] = feedback_data
        
        # TODO: Submit to LangSmith if available
        langsmith_submitted = False
        
        logger.info(f"✅ Feedback submitted successfully: {feedback_id}")
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="submitted",
            langsmith_submitted=langsmith_submitted,
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.post("/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback_alias(request: FeedbackRequest):
    """Submit user feedback for a report (alias endpoint for frontend compatibility)"""
    return await submit_feedback(request)

@router.get("/feedback")
async def list_feedback():
    """List all feedback entries"""
    try:
        feedback_list = list(feedback_storage.values())
        return sorted(feedback_list, key=lambda x: x.get("created_at", datetime.now()), reverse=True)
        
    except Exception as e:
        logger.error(f"❌ Error listing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list feedback: {str(e)}")

@router.get("/feedback/{feedback_id}")
async def get_feedback(feedback_id: str):
    """Get specific feedback by ID"""
    try:
        if feedback_id not in feedback_storage:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return feedback_storage[feedback_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting feedback {feedback_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")

@router.get("/reports/{report_id}/feedback")
async def get_report_feedback(report_id: str):
    """Get all feedback for a specific report"""
    try:
        report_feedback = [
            feedback for feedback in feedback_storage.values()
            if feedback.get("report_id") == report_id
        ]
        
        return sorted(report_feedback, key=lambda x: x.get("created_at", datetime.now()), reverse=True)
        
    except Exception as e:
        logger.error(f"❌ Error getting feedback for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report feedback: {str(e)}")

# Analytics Endpoints

@router.get("/feedback/analytics/overview")
async def get_feedback_overview():
    """Get feedback analytics overview"""
    try:
        all_feedback = list(feedback_storage.values())

        total_feedback = len(all_feedback)
        thumbs_up_count = len([f for f in all_feedback if f.get("feedback_type") == "thumbs_up"])
        thumbs_down_count = len([f for f in all_feedback if f.get("feedback_type") == "thumbs_down"])

        # Calculate feedback rate
        feedback_rate = round((thumbs_up_count / total_feedback * 100) if total_feedback > 0 else 0, 1)

        # Count unique reports and sessions
        unique_reports = len(set(f.get("report_id") for f in all_feedback if f.get("report_id")))
        unique_sessions = len(set(f.get("user_session_id") for f in all_feedback if f.get("user_session_id")))

        return {
            "total_feedback": total_feedback,
            "thumbs_up_count": thumbs_up_count,
            "thumbs_down_count": thumbs_down_count,
            "feedback_rate": feedback_rate,
            "unique_reports": unique_reports,
            "unique_sessions": unique_sessions
        }

    except Exception as e:
        logger.error(f"❌ Error getting feedback overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback overview: {str(e)}")

@router.get("/feedback/analytics/trends")
async def get_feedback_trends(days: int = Query(7, ge=1, le=90)):
    """Get feedback trends over specified number of days"""
    try:
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        # Get all feedback
        all_feedback = list(feedback_storage.values())

        # Group feedback by date
        daily_stats = {}
        for i in range(days):
            date = start_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily_stats[date_str] = {
                "date": date_str,
                "thumbs_up": 0,
                "thumbs_down": 0,
                "total": 0
            }

        # Process feedback data
        for feedback in all_feedback:
            created_at = feedback.get("created_at")
            if not created_at:
                continue

            # Handle datetime objects
            if isinstance(created_at, datetime):
                feedback_date = created_at
            else:
                # Parse string datetime
                try:
                    feedback_date = datetime.fromisoformat(str(created_at))
                except:
                    continue

            # Ensure timezone-aware comparison
            if feedback_date.tzinfo is None:
                feedback_date = feedback_date.replace(tzinfo=timezone.utc)

            if start_date <= feedback_date <= end_date:
                date_str = feedback_date.strftime("%Y-%m-%d")
                if date_str in daily_stats:
                    daily_stats[date_str]["total"] += 1
                    if feedback.get("feedback_type") == "thumbs_up":
                        daily_stats[date_str]["thumbs_up"] += 1
                    else:
                        daily_stats[date_str]["thumbs_down"] += 1

        return {
            "period_days": days,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily_stats": list(daily_stats.values())
        }

    except Exception as e:
        logger.error(f"❌ Error getting feedback trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback trends: {str(e)}")

@router.get("/feedback/analytics/reports")
async def get_feedback_by_reports(limit: int = Query(10, ge=1, le=50)):
    """Get feedback analytics grouped by reports"""
    try:
        all_feedback = list(feedback_storage.values())

        # Group feedback by report_id
        report_stats = defaultdict(lambda: {"thumbs_up": 0, "thumbs_down": 0, "total": 0})

        for feedback in all_feedback:
            report_id = feedback.get("report_id")
            if not report_id:
                continue

            report_stats[report_id]["total"] += 1
            if feedback.get("feedback_type") == "thumbs_up":
                report_stats[report_id]["thumbs_up"] += 1
            else:
                report_stats[report_id]["thumbs_down"] += 1

        # Convert to list and sort by total feedback
        reports_list = []
        for report_id, stats in report_stats.items():
            reports_list.append({
                "report_id": report_id,
                "thumbs_up": stats["thumbs_up"],
                "thumbs_down": stats["thumbs_down"],
                "total": stats["total"]
            })

        # Sort by total feedback (descending) and limit
        reports_list.sort(key=lambda x: x["total"], reverse=True)

        return {
            "reports": reports_list[:limit],
            "total_reports_with_feedback": len(reports_list)
        }

    except Exception as e:
        logger.error(f"❌ Error getting feedback by reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback by reports: {str(e)}")

@router.get("/feedback/analytics/recent")
async def get_recent_feedback(limit: int = Query(20, ge=1, le=100)):
    """Get recent feedback entries"""
    try:
        all_feedback = list(feedback_storage.values())

        # Sort by created_at (most recent first)
        sorted_feedback = sorted(
            all_feedback,
            key=lambda x: x.get("created_at", datetime.min),
            reverse=True
        )

        # Format for frontend display
        recent_feedback = []
        for feedback in sorted_feedback[:limit]:
            recent_feedback.append({
                "feedback_id": feedback.get("feedback_id"),
                "report_id": feedback.get("report_id"),
                "feedback_type": feedback.get("feedback_type"),
                "rating": feedback.get("rating"),
                "comment": feedback.get("comment", "")[:100] + "..." if feedback.get("comment") and len(feedback.get("comment", "")) > 100 else feedback.get("comment", ""),
                "user_session_id": feedback.get("user_session_id", "")[:12] + "..." if feedback.get("user_session_id") else None,
                "created_at": feedback.get("created_at").isoformat() if isinstance(feedback.get("created_at"), datetime) else str(feedback.get("created_at")),
                "source": feedback.get("source", "web")
            })

        return {
            "recent_feedback": recent_feedback,
            "total_count": len(all_feedback)
        }

    except Exception as e:
        logger.error(f"❌ Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent feedback: {str(e)}")

@router.get("/feedback/analytics/export")
async def export_feedback_analytics(
    format: str = Query("json", pattern="^(json|csv)$"),
    days: int = Query(30, ge=1, le=365)
):
    """Export feedback analytics data"""
    try:
        # Get trends data for the specified period
        trends_response = await get_feedback_trends(days)
        overview_response = await get_feedback_overview()

        export_data = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "overview": overview_response,
            "trends": trends_response,
            "metadata": {
                "total_feedback_exported": overview_response["total_feedback"],
                "export_format": format
            }
        }

        if format == "json":
            return export_data
        else:
            # For CSV format, return a simplified structure
            # In a real implementation, you'd generate actual CSV content
            return {
                "message": "CSV export functionality would be implemented here",
                "data": export_data
            }

    except Exception as e:
        logger.error(f"❌ Error exporting feedback analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export feedback analytics: {str(e)}")
