#!/usr/bin/env python3
"""
Health check router for monitoring application status.
"""

import logging
from datetime import datetime

from fastapi import APIRouter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import HealthResponse
from config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        services_status = {
            "cache": settings.CACHE_AVAILABLE,
            "database": settings.DATABASE_AVAILABLE,
            "message_broker": settings.RABBITMQ_AVAILABLE,
            "financial_metrics_agent": True  # Will be checked during import
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat()
        )

@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AgentInvest Financial Analyzer API",
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }
