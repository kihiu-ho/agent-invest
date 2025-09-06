"""
Base Agent Infrastructure for Search Agents

This module provides the base classes and infrastructure for search agents
used in the AgentInvest system.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    REQUEST = "request"
    RESPONSE = "response"


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class Message:
    """Message structure for agent communication."""
    type: MessageType
    content: str
    sender: str
    recipient: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """
    Base class for all search agents in the AgentInvest system.
    
    Provides common functionality for:
    - Logging
    - Status management
    - Message handling
    - Error handling
    """
    
    def __init__(self, name: str = None):
        """Initialize the base agent."""
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        self.status = AgentStatus.IDLE
        self.messages: List[Message] = []
        self.start_time = None
        self.end_time = None
        
    def set_status(self, status: AgentStatus, message: str = None):
        """Set the agent status and optionally log a message."""
        self.status = status
        if message:
            self.logger.info(f"Status changed to {status.value}: {message}")
            self.add_message(MessageType.INFO, message)
    
    def add_message(self, msg_type: MessageType, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the agent's message log."""
        message = Message(
            type=msg_type,
            content=content,
            sender=self.name,
            metadata=metadata or {}
        )
        self.messages.append(message)
        
        # Log the message
        if msg_type == MessageType.ERROR:
            self.logger.error(content)
        elif msg_type == MessageType.WARNING:
            self.logger.warning(content)
        else:
            self.logger.info(content)
    
    def get_messages(self, msg_type: MessageType = None) -> List[Message]:
        """Get messages, optionally filtered by type."""
        if msg_type is None:
            return self.messages.copy()
        return [msg for msg in self.messages if msg.type == msg_type]
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages.clear()
    
    def start_work(self):
        """Mark the start of work."""
        self.start_time = datetime.now()
        self.set_status(AgentStatus.WORKING, "Starting work")
    
    def complete_work(self, success: bool = True):
        """Mark the completion of work."""
        self.end_time = datetime.now()
        if success:
            self.set_status(AgentStatus.COMPLETED, "Work completed successfully")
        else:
            self.set_status(AgentStatus.ERROR, "Work completed with errors")
    
    def get_work_duration(self) -> Optional[float]:
        """Get the duration of work in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Abstract method that must be implemented by subclasses.
        
        This method should contain the main logic of the agent.
        """
        pass
    
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the agent with proper status management and error handling.
        """
        try:
            self.start_work()
            result = await self.process(*args, **kwargs)
            self.complete_work(success=True)
            return result
        except Exception as e:
            self.add_message(MessageType.ERROR, f"Error during processing: {str(e)}")
            self.complete_work(success=False)
            raise
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's current status."""
        return {
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.get_work_duration(),
            "message_count": len(self.messages),
            "error_count": len(self.get_messages(MessageType.ERROR)),
            "warning_count": len(self.get_messages(MessageType.WARNING))
        }


class SearchAgent(BaseAgent):
    """
    Specialized base class for search agents.
    
    Provides additional functionality specific to search operations.
    """
    
    def __init__(self, name: str = None, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the search agent."""
        super().__init__(name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.search_results = []
    
    async def retry_operation(self, operation, *args, **kwargs):
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: The async function to retry
            *args, **kwargs: Arguments to pass to the operation
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting operation (attempt {attempt + 1}/{self.max_retries})")
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    self.add_message(MessageType.SUCCESS, f"Operation succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_exception = e
                self.add_message(MessageType.WARNING, f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries failed
        self.add_message(MessageType.ERROR, f"All {self.max_retries} attempts failed")
        raise last_exception
    
    def add_search_result(self, result: Dict[str, Any]):
        """Add a search result to the results list."""
        self.search_results.append(result)
        self.add_message(MessageType.INFO, f"Added search result: {result.get('title', 'Unknown')}")
    
    def get_search_results(self) -> List[Dict[str, Any]]:
        """Get all search results."""
        return self.search_results.copy()
    
    def clear_search_results(self):
        """Clear all search results."""
        self.search_results.clear()
        self.add_message(MessageType.INFO, "Cleared search results")


# Convenience functions for creating common message types
def create_info_message(content: str, sender: str, metadata: Dict[str, Any] = None) -> Message:
    """Create an info message."""
    return Message(MessageType.INFO, content, sender, metadata=metadata)


def create_error_message(content: str, sender: str, metadata: Dict[str, Any] = None) -> Message:
    """Create an error message."""
    return Message(MessageType.ERROR, content, sender, metadata=metadata)


def create_warning_message(content: str, sender: str, metadata: Dict[str, Any] = None) -> Message:
    """Create a warning message."""
    return Message(MessageType.WARNING, content, sender, metadata=metadata)


def create_success_message(content: str, sender: str, metadata: Dict[str, Any] = None) -> Message:
    """Create a success message."""
    return Message(MessageType.SUCCESS, content, sender, metadata=metadata)
