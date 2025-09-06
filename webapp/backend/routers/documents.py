#!/usr/bin/env python3
"""
Documents router for handling document processing and querying endpoints.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DocumentRequest, DocumentResponse, QueryRequest, QueryResponse
from services.document_service import DocumentService

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["documents"])

# Global service instance (will be injected via dependency)
_document_service: Optional[DocumentService] = None

def get_document_service() -> DocumentService:
    """Dependency to get document service instance"""
    global _document_service
    if _document_service is None:
        raise HTTPException(status_code=500, detail="Document service not initialized")
    return _document_service

def set_document_service(service: DocumentService):
    """Set the global document service instance"""
    global _document_service
    _document_service = service

@router.post("/documents/process", response_model=DocumentResponse)
async def process_documents(
    request: DocumentRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """Process documents for a ticker"""
    try:
        logger.info(f"📄 Processing documents for ticker: {request.ticker}")
        
        result = await document_service.download_and_process_documents(
            ticker=request.ticker,
            document_type=request.document_type,
            year=request.year
        )
        
        if result.get("success", False):
            return DocumentResponse(
                operation_id=result["operation_id"],
                ticker=result["ticker"],
                status=result["status"],
                document_count=result["document_count"],
                processing_time=result["processing_time"]
            )
        else:
            return DocumentResponse(
                operation_id=result.get("operation_id", ""),
                ticker=request.ticker,
                status="failed",
                document_count=0,
                error_message=result.get("error", "Unknown error")
            )
        
    except Exception as e:
        logger.error(f"❌ Error processing documents for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

@router.post("/documents/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """Query processed documents"""
    try:
        logger.info(f"🔍 Querying documents: {request.query}")
        
        result = await document_service.query_documents(
            query=request.query,
            ticker=request.ticker,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        if result.get("success", False):
            return QueryResponse(
                query_id=result["query_id"],
                results=result["results"],
                total_results=result["total_results"],
                processing_time=result["processing_time"]
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Query failed"))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query documents: {str(e)}")

@router.get("/documents/operations")
async def list_operations(document_service: DocumentService = Depends(get_document_service)):
    """List all document operations"""
    try:
        operations = document_service.list_operations()
        return operations
        
    except Exception as e:
        logger.error(f"❌ Error listing operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list operations: {str(e)}")

@router.get("/documents/operations/{operation_id}")
async def get_operation(
    operation_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get specific operation by ID"""
    try:
        operation = document_service.get_operation(operation_id)
        
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        
        return operation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting operation {operation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get operation: {str(e)}")

@router.get("/documents/{ticker}")
async def get_ticker_documents(
    ticker: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get processed documents for a specific ticker"""
    try:
        documents = document_service.get_processed_documents(ticker)
        return {
            "ticker": ticker,
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting documents for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")
