#!/usr/bin/env python3
"""
AgentInvest P2 Backend Startup Script
Standalone version of the AgentInvest HTML report generation system
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the Python path and environment variables"""
    # Get the current directory (p2 folder)
    p2_dir = Path(__file__).parent.absolute()
    
    # Add the financial_metrics_agent to Python path
    financial_agent_path = p2_dir / "financial_metrics_agent"
    if str(financial_agent_path) not in sys.path:
        sys.path.insert(0, str(financial_agent_path))
    
    # Add the backend directory to Python path
    backend_path = p2_dir / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = f"{financial_agent_path}:{backend_path}:{os.environ.get('PYTHONPATH', '')}"
    
    print(f"✅ Python path configured:")
    print(f"   - Financial Agent: {financial_agent_path}")
    print(f"   - Backend: {backend_path}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import yfinance
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    setup_environment()
    
    if not check_dependencies():
        return
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    print("🚀 Starting AgentInvest P2 Backend Server...")
    print("📊 Server will be available at: http://localhost:8001")
    print("📋 API Documentation: http://localhost:8001/docs")
    print("🔄 Press Ctrl+C to stop the server")

    # Start the server
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()
