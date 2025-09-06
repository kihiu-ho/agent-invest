#!/usr/bin/env python3
"""
WebSocket router for real-time report progress updates.
"""

import logging
import json
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["websocket"])

# Global connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, report_id: str):
        """Accept WebSocket connection and add to report-specific group"""
        await websocket.accept()
        
        if report_id not in self.active_connections:
            self.active_connections[report_id] = set()
        
        self.active_connections[report_id].add(websocket)
        logger.info(f"🔌 WebSocket connected for report {report_id}. Total connections: {len(self.active_connections[report_id])}")

    def disconnect(self, websocket: WebSocket, report_id: str):
        """Remove WebSocket connection from report-specific group"""
        if report_id in self.active_connections:
            self.active_connections[report_id].discard(websocket)
            
            # Clean up empty groups
            if not self.active_connections[report_id]:
                del self.active_connections[report_id]
                
        logger.info(f"🔌 WebSocket disconnected for report {report_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")

    async def send_to_report(self, report_id: str, message: dict):
        """Send message to all WebSockets connected to a specific report"""
        if report_id not in self.active_connections:
            logger.debug(f"No active connections for report {report_id}")
            return

        message_text = json.dumps(message)
        disconnected_sockets = set()
        
        for websocket in self.active_connections[report_id].copy():
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(message_text)
                else:
                    disconnected_sockets.add(websocket)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                disconnected_sockets.add(websocket)
        
        # Clean up disconnected sockets
        for websocket in disconnected_sockets:
            self.active_connections[report_id].discard(websocket)

# Global connection manager instance
manager = ConnectionManager()

@router.websocket("/ws/reports/{report_id}")
async def websocket_endpoint(websocket: WebSocket, report_id: str):
    """WebSocket endpoint for real-time report progress updates"""
    await manager.connect(websocket, report_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "report_id": report_id,
                "message": f"Connected to report {report_id} updates"
            }),
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (ping/pong, etc.)
                data = await websocket.receive_text()
                
                # Echo back ping messages for connection health
                if data == "ping":
                    await manager.send_personal_message("pong", websocket)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for report {report_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for report {report_id}")
    except Exception as e:
        logger.error(f"WebSocket error for report {report_id}: {e}")
    finally:
        manager.disconnect(websocket, report_id)

# Function to send updates to WebSocket clients (used by report consumer)
async def send_websocket_update(report_id: str, status: str, progress: int, message: str):
    """Send progress update to all WebSocket clients for a specific report"""
    update_message = {
        "type": "progress",
        "report_id": report_id,
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": json.dumps({"$date": {"$numberLong": str(int(__import__('time').time() * 1000))}})
    }
    
    await manager.send_to_report(report_id, update_message)
    logger.info(f"📡 Sent WebSocket update for report {report_id}: {status} ({progress}%)")

# Export the connection manager and update function for use by other modules
__all__ = ["router", "manager", "send_websocket_update"]
