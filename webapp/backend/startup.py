#!/usr/bin/env python3
"""
Startup module for initializing services and dependencies.
"""

import logging
from typing import Optional

from config import settings
from services.report_service import ReportService
from services.document_service import DocumentService
from routers.reports import set_report_service
from routers.documents import set_document_service

# Setup logging
logger = logging.getLogger(__name__)

# Global service instances
cache_service: Optional[object] = None
database_service: Optional[object] = None
message_broker: Optional[object] = None
report_service: Optional[ReportService] = None
document_service: Optional[DocumentService] = None

def initialize_cache_service():
    """Initialize cache service if available"""
    global cache_service
    
    if not settings.CACHE_AVAILABLE:
        logger.info("⚠️ Cache service disabled")
        return None
    
    try:
        from services.cache_service import CacheService
        cache_service = CacheService()
        logger.info("✅ Cache service initialized")
        return cache_service
    except ImportError:
        logger.warning("⚠️ Cache service not available")
        return None

def initialize_database_service():
    """Initialize database service if available"""
    global database_service
    
    if not settings.DATABASE_AVAILABLE or not settings.USE_DATABASE:
        logger.info("⚠️ Database service disabled")
        return None
    
    try:
        from services.database_service import DatabaseService
        database_service = DatabaseService()
        logger.info("✅ Database service initialized")
        return database_service
    except ImportError:
        logger.warning("⚠️ Database service not available")
        return None

def initialize_message_broker():
    """Initialize message broker if available"""
    global message_broker
    
    if not settings.RABBITMQ_AVAILABLE or not settings.USE_RABBITMQ:
        logger.info("⚠️ Message broker service disabled")
        return None
    
    try:
        from services.message_broker import MessageBroker
        message_broker = MessageBroker()
        logger.info("✅ Message broker service initialized")
        return message_broker
    except ImportError:
        logger.warning("⚠️ Message broker service not available")
        return None

def initialize_document_service():
    """Initialize document service"""
    global document_service

    # Initialize dependencies
    cache = cache_service
    database = database_service

    # Create document service
    document_service = DocumentService(
        cache_service=cache,
        database_service=database
    )

    # Set the global document service for dependency injection
    set_document_service(document_service)

    logger.info("✅ Document service initialized")
    return document_service

def initialize_report_service():
    """Initialize report service"""
    global report_service

    # Initialize dependencies
    cache = initialize_cache_service()
    database = initialize_database_service()
    broker = initialize_message_broker()

    # Create report service
    report_service = ReportService(
        cache_service=cache,
        database_service=database,
        message_broker=broker
    )

    # Set the global report service for dependency injection
    set_report_service(report_service)

    logger.info("✅ Report service initialized")
    return report_service

async def startup_event():
    """Application startup event handler"""
    logger.info("🚀 Starting AgentInvest Financial Analyzer API...")

    # Initialize all services
    initialize_report_service()
    initialize_document_service()

    logger.info("✅ Application startup completed")

async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("🛑 Shutting down AgentInvest Financial Analyzer API...")

    # Cleanup services if needed
    global cache_service, database_service, message_broker, report_service

    # Cleanup enhanced orchestrator
    if report_service:
        try:
            await report_service.cleanup()
            logger.info("✅ Report service cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error cleaning up report service: {e}")

    # Close connections, cleanup resources, etc.
    if database_service:
        try:
            # database_service.close()
            pass
        except Exception as e:
            logger.error(f"Error closing database service: {e}")

    if message_broker:
        try:
            # message_broker.close()
            pass
        except Exception as e:
            logger.error(f"Error closing message broker: {e}")

    logger.info("✅ Application shutdown completed")
