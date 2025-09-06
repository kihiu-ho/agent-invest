#!/usr/bin/env python3
"""
Configuration module for the FastAPI backend application.
Handles environment variables, logging setup, and application settings.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configuration"""
    
    # Application info
    APP_NAME: str = "AgentInvest Financial Analyzer API"
    APP_DESCRIPTION: str = "REST API for generating comprehensive financial analysis reports"
    APP_VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS settings - can be overridden by environment variable
    _default_cors_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:30080",
        "http://127.0.0.1:30080",
        "http://localhost:30081",
        "http://127.0.0.1:30081",
        "http://webapp-frontend-service",
        "http://webapp-frontend-service.webapp.svc.cluster.local",
        "http://webapp-frontend-service.functorhk.svc.cluster.local",
        "https://agentinvest.applenova.store:30443",
        "https://45.77.206.161:30443",
        "https://agentinvest.applenova.store",
        "https://45.77.206.161",
        "http://45.77.206.161:30084",  # Current frontend deployment
        "http://45.77.206.161:30083",  # Current backend deployment
        "ws://localhost:3001",
        "ws://127.0.0.1:3001",
        "wss://agentinvest.applenova.store:30443",
        "wss://45.77.206.161:30443",
        "wss://agentinvest.applenova.store",
        "wss://45.77.206.161",
        "null"  # Allow file:// protocol for development
    ]

    # Allow CORS origins to be overridden by environment variable
    cors_env = os.getenv("CORS_ORIGINS")
    CORS_ORIGINS: List[str] = cors_env.split(",") if cors_env else _default_cors_origins
    
    # Service availability flags
    CACHE_AVAILABLE: bool = False
    DATABASE_AVAILABLE: bool = False
    RABBITMQ_AVAILABLE: bool = False
    
    # Feature flags
    USE_DATABASE: bool = os.getenv('USE_DATABASE', 'false').lower() == 'true'
    USE_RABBITMQ: bool = os.getenv('USE_RABBITMQ', 'false').lower() == 'true'
    USE_ASYNC_REPORTS: bool = os.getenv('USE_ASYNC_REPORTS', 'true').lower() == 'true'
    
    # Paths - handle both local and container environments
    REPORTS_DIR: Path = Path("./reports") if not Path("/app").exists() else Path("/app/reports")
    LOGS_DIR: Path = Path("./logs") if not Path("/app").exists() else Path("/app/logs")
    
    def __init__(self):
        """Initialize settings and check service availability"""
        self._check_service_availability()
        self._ensure_directories()
    
    def _check_service_availability(self):
        """Check which services are available"""
        try:
            from services.cache_service import CacheService
            self.CACHE_AVAILABLE = True
            logger.info("✅ Cache service available")
        except ImportError:
            logger.warning("⚠️ Cache service not available - caching disabled")
        
        try:
            from services.database_service import DatabaseService
            self.DATABASE_AVAILABLE = True
            logger.info("✅ Database service available")
        except ImportError:
            logger.warning("⚠️ Database service not available - using in-memory storage")
        
        try:
            from services.message_broker import MessageBroker
            self.RABBITMQ_AVAILABLE = True
            logger.info("✅ Message broker service available")
        except ImportError:
            logger.warning("⚠️ Message broker service not available - async messaging disabled")
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
