#!/usr/bin/env python3
"""
Pydantic models for the FastAPI backend application.
Defines request/response models and data validation.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator

class ReportRequest(BaseModel):
    """Request model for report generation"""
    ticker: str
    
    @field_validator('ticker')
    @classmethod
    def validate_hk_ticker(cls, v):
        """Validate Hong Kong stock ticker format (XXXX.HK)"""
        # Allow 1-5 digits followed by .HK (will be normalized to 4 digits)
        pattern = r'^\d{1,5}\.HK$'
        if not re.match(pattern, v.upper()):
            raise ValueError('Ticker must be in format XXXX.HK where XXXX is 1-5 digits')

        # Normalize to 4-digit format (pad with leading zeros)
        ticker_parts = v.upper().split('.')
        ticker_code = ticker_parts[0].zfill(4)  # Pad to 4 digits
        return f"{ticker_code}.HK"

class ReportResponse(BaseModel):
    """Response model for report data"""
    report_id: str
    ticker: str
    status: str
    created_at: datetime
    report_file: Optional[str] = None
    charts_generated: Optional[int] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class ReportStatus(BaseModel):
    """Model for report generation status"""
    report_id: str
    status: str
    progress: int
    message: str
    created_at: datetime

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    report_id: str
    feedback_type: str  # "thumbs_up" or "thumbs_down"
    category: str = "overall_quality"
    rating: Optional[int] = None
    comment: Optional[str] = None
    user_session_id: Optional[str] = None
    feedback_context: Optional[Dict[str, Any]] = None

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str
    status: str
    langsmith_submitted: bool
    message: str

class DocumentRequest(BaseModel):
    """Request model for document processing"""
    ticker: str
    document_type: str = "annual_report"
    year: Optional[int] = None

class DocumentResponse(BaseModel):
    """Response model for document processing"""
    operation_id: str
    ticker: str
    status: str
    document_count: int
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

class QueryRequest(BaseModel):
    """Request model for document queries"""
    query: str
    ticker: Optional[str] = None
    limit: int = 10
    similarity_threshold: float = 0.7

class QueryResponse(BaseModel):
    """Response model for document queries"""
    query_id: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    services: Optional[Dict[str, bool]] = None
