#!/usr/bin/env python3
"""
FastAPI backend for AgentInvest - Financial Analysis and Report Generation
Modular architecture with financial_metrics_agent integration
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from startup import startup_event, shutdown_event
from routers import reports, feedback, health, documents, websocket

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Register event handlers
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Include routers
app.include_router(health.router)
app.include_router(reports.router)
app.include_router(feedback.router)
app.include_router(documents.router)
app.include_router(websocket.router)

# Main entry point
if __name__ == "__main__":
    logger.info("🚀 Starting AgentInvest Financial Analyzer API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
