#!/usr/bin/env python3
"""
Reports router for handling report generation and management endpoints.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ReportRequest, ReportResponse, ReportStatus
from services.report_service import ReportService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["reports"])

# Global service instances (will be injected via dependency)
_report_service: Optional[ReportService] = None

def get_report_service() -> ReportService:
    """Dependency to get report service instance"""
    global _report_service
    if _report_service is None:
        raise HTTPException(status_code=500, detail="Report service not initialized")
    return _report_service

def set_report_service(service: ReportService):
    """Set the global report service instance"""
    global _report_service
    _report_service = service

async def process_report_async(report_id: str, ticker: str, report_service: ReportService):
    """Background task to process report generation"""
    try:
        logger.info(f"🚀 Starting async report generation for {ticker} (ID: {report_id})")
        
        # Update status to processing
        if report_id in report_service.report_tasks:
            report_service.report_tasks[report_id].update({
                "status": "processing",
                "progress": 10,
                "message": "Starting analysis..."
            })
        
        # Generate the enhanced report
        result = await report_service.generate_enhanced_report(ticker)
        
        # Update task with results
        if report_id in report_service.report_tasks:
            if result.get("success", False):
                report_service.report_tasks[report_id].update({
                    "status": "completed",
                    "progress": 100,
                    "message": "Report generation completed",
                    "completed_at": datetime.now(),
                    "report_file": result.get("report_file"),
                    "session_directory": result.get("session_directory"),
                    "charts_generated": result.get("charts_generated", 0),
                    "processing_time": result.get("processing_time", 0),
                    "analysis_result": result.get("analysis_result", {}),
                    "quality_score": result.get("quality_score", 0.8)
                })
                
                # Cache the successful report
                await report_service.cache_report(ticker, report_service.report_tasks[report_id])
                
                logger.info(f"✅ Report generation completed for {ticker} (ID: {report_id})")
            else:
                report_service.report_tasks[report_id].update({
                    "status": "failed",
                    "progress": 0,
                    "message": f"Report generation failed: {result.get('error', 'Unknown error')}",
                    "error_message": result.get("error", "Unknown error"),
                    "completed_at": datetime.now()
                })
                logger.error(f"❌ Report generation failed for {ticker} (ID: {report_id}): {result.get('error')}")
        
    except Exception as e:
        logger.error(f"❌ Async report processing failed for {ticker} (ID: {report_id}): {e}")
        if report_id in report_service.report_tasks:
            report_service.report_tasks[report_id].update({
                "status": "failed",
                "progress": 0,
                "message": f"Processing failed: {str(e)}",
                "error_message": str(e),
                "completed_at": datetime.now()
            })

@router.post("/reports", response_model=ReportResponse)
async def create_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    report_service: ReportService = Depends(get_report_service)
):
    """Create a new financial analysis report"""
    try:
        logger.info(f"📊 Creating report for ticker: {request.ticker}")
        
        # Create the report task
        result = await report_service.create_report(request.ticker)
        
        # If not cached, start background processing
        if not result.get("cached", False):
            background_tasks.add_task(
                process_report_async,
                result["report_id"],
                request.ticker,
                report_service
            )
        
        return ReportResponse(
            report_id=result["report_id"],
            ticker=result["ticker"],
            status=result["status"],
            created_at=result["created_at"]
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating report for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")

@router.get("/reports", response_model=List[ReportResponse])
async def list_reports(report_service: ReportService = Depends(get_report_service)):
    """List all reports"""
    try:
        reports = report_service.list_reports()
        
        return [
            ReportResponse(
                report_id=report.get("report_id", ""),
                ticker=report.get("ticker", ""),
                status=report.get("status", "unknown"),
                created_at=report.get("created_at", datetime.now()),
                report_file=report.get("report_file"),
                charts_generated=report.get("charts_generated"),
                processing_time=report.get("processing_time"),
                error_message=report.get("error_message")
            )
            for report in reports
        ]
        
    except Exception as e:
        logger.error(f"❌ Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")

@router.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """Get a specific report by ID"""
    try:
        report = report_service.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return ReportResponse(
            report_id=report.get("report_id", ""),
            ticker=report.get("ticker", ""),
            status=report.get("status", "unknown"),
            created_at=report.get("created_at", datetime.now()),
            report_file=report.get("report_file"),
            charts_generated=report.get("charts_generated"),
            processing_time=report.get("processing_time"),
            error_message=report.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")

@router.get("/reports/{report_id}/status", response_model=ReportStatus)
async def get_report_status(
    report_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """Get the status of a report generation task"""
    try:
        report = report_service.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return ReportStatus(
            report_id=report.get("report_id", ""),
            status=report.get("status", "unknown"),
            progress=report.get("progress", 0),
            message=report.get("message", ""),
            created_at=report.get("created_at", datetime.now())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting report status {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report status: {str(e)}")

@router.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """Delete a report"""
    try:
        report = report_service.get_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Remove from both storages
        if report_id in report_service.report_tasks:
            del report_service.report_tasks[report_id]
        
        if report_id in report_service.async_report_queue:
            del report_service.async_report_queue[report_id]
        
        return {"message": "Report deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")

@router.get("/reports/{report_id}/file")
async def get_report_file(
    report_id: str,
    report_service: ReportService = Depends(get_report_service)
):
    """Serve the generated report HTML file"""
    try:
        logger.info(f"📄 Serving report file for ID: {report_id}")

        # Get the report details
        report = report_service.get_report(report_id)

        if not report:
            logger.warning(f"❌ Report not found: {report_id}")
            raise HTTPException(status_code=404, detail="Report not found")

        # Check if report is completed and has a file
        if report.get("status") != "completed":
            logger.warning(f"⏳ Report not completed yet: {report_id} (status: {report.get('status')})")
            raise HTTPException(
                status_code=202,
                detail=f"Report is still {report.get('status', 'processing')}. Please try again later."
            )

        report_file = report.get("report_file")
        if not report_file:
            logger.warning(f"❌ No report file found for: {report_id}")
            raise HTTPException(status_code=404, detail="Report file not found")

        # Convert to Path object and check if file exists
        file_path = Path(report_file)
        if not file_path.exists():
            logger.warning(f"❌ Report file does not exist: {file_path}")
            raise HTTPException(status_code=404, detail="Report file not found on disk")

        # Check if it's an HTML file
        if file_path.suffix.lower() == '.html':
            logger.info(f"✅ Serving HTML report file: {file_path}")
            # Read and return HTML content directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content)
            except Exception as e:
                logger.error(f"❌ Error reading HTML file {file_path}: {e}")
                raise HTTPException(status_code=500, detail="Error reading report file")
        else:
            # For other file types, use FileResponse
            logger.info(f"✅ Serving file: {file_path}")
            return FileResponse(
                path=str(file_path),
                filename=f"report_{report.get('ticker', 'unknown')}_{report_id}.{file_path.suffix}",
                media_type='application/octet-stream'
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving report file {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve report file: {str(e)}")

@router.get("/orchestrator/status")
async def get_orchestrator_status(report_service: ReportService = Depends(get_report_service)):
    """Get comprehensive status of the orchestrator and its enhanced components"""
    try:
        status = await report_service.get_orchestrator_status()
        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"❌ Error getting orchestrator status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get orchestrator status: {str(e)}")

@router.post("/cache/invalidate/{ticker}")
async def invalidate_cache(
    ticker: str,
    report_service: ReportService = Depends(get_report_service)
):
    """Invalidate cache for a specific ticker"""
    try:
        result = await report_service.invalidate_cache(ticker)

        if result.get("success", False):
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Cache invalidation failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error invalidating cache for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")

@router.get("/cache/stats")
async def get_cache_statistics(report_service: ReportService = Depends(get_report_service)):
    """Get cache performance statistics"""
    try:
        status = await report_service.get_orchestrator_status()

        cache_stats = {}
        if status.get("available", False):
            enhanced_components = status.get("enhanced_components", {})
            cache_manager = enhanced_components.get("cache_manager", {})
            cache_stats = cache_manager.get("stats", {})

        return JSONResponse(content={
            "cache_available": bool(cache_stats),
            "statistics": cache_stats
        })

    except Exception as e:
        logger.error(f"❌ Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")
