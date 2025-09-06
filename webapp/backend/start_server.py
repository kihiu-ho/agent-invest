#!/usr/bin/env python3
"""
Startup script for Enhanced AutoGen Financial Analyzer Backend

This script starts the FastAPI backend server with proper configuration
and environment setup.
"""

import os
import sys
import logging
from pathlib import Path

# Add the enhanced_autogen_refactored to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import fastapi
        import uvicorn
        import websockets
        import pydantic
        logger.info("✅ FastAPI dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing FastAPI dependency: {e}")
        logger.error("Please install: pip install -r requirements.txt")
        return False
    
    try:
        # Check if financial_metrics_agent is available instead
        import sys
        sys.path.append(str(project_root / "financial_metrics_agent"))
        from orchestrator import FinancialMetricsOrchestrator
        logger.info("✅ Financial Metrics Agent system available")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Financial Metrics Agent not available: {e}")
        logger.info("🔄 Continuing with basic API functionality")
        return True  # Allow server to start without the agent

def main():
    """Main startup function"""
    logger.info("🚀 Starting Enhanced AutoGen Financial Analyzer Backend")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", str(project_root))
    
    # Import and run the server
    try:
        import uvicorn
        from main import app
        
        logger.info("🌐 Starting FastAPI server...")
        logger.info("📊 Backend will be available at: http://localhost:8000")
        logger.info("📚 API documentation at: http://localhost:8000/docs")
        logger.info("🔌 WebSocket endpoint: ws://localhost:8000/ws/reports/{report_id}")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
