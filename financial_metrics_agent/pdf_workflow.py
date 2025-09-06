#!/usr/bin/env python3
"""
PDF Workflow Manager for Financial Documents

Handles PDF document processing, chunking, and embedding generation
for HKEX annual reports and financial documents.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing workflow."""
    enable_verification: bool = True
    enable_download: bool = True
    enable_chunking: bool = True
    enable_embeddings: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    max_file_size_mb: int = 50
    download_timeout: int = 300

@dataclass
class PDFWorkflowResult:
    """Result of PDF workflow execution."""
    step: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    files_processed: int = 0

class PDFWorkflowManager:
    """
    Manages the PDF document processing workflow for financial analysis.
    
    Handles:
    - PDF document verification
    - HKEX annual report downloading
    - Document chunking and text extraction
    - Embedding generation for vector storage
    """
    
    def __init__(self, config: Optional[PDFProcessingConfig] = None,
                 hk_data_downloader=None, pdf_processor=None, vector_store=None):
        """
        Initialize the PDF workflow manager.
        
        Args:
            config: PDF processing configuration
            hk_data_downloader: HK data downloader instance
            pdf_processor: PDF processor instance
            vector_store: Vector store instance
        """
        self.config = config or PDFProcessingConfig()
        self.hk_data_downloader = hk_data_downloader
        self.pdf_processor = pdf_processor
        self.vector_store = vector_store
        
        # Track processing statistics
        self.processed_documents = 0
        self.generated_embeddings = 0
        self.processing_errors = 0
        
        logger.info("✅ PDF workflow manager initialized")
    
    async def execute_complete_workflow(self, ticker: str) -> Dict[str, Any]:
        """
        Execute the complete PDF processing workflow for a ticker.
        
        Args:
            ticker: HK ticker symbol
            
        Returns:
            Complete workflow results
        """
        logger.info(f"📄 Starting complete PDF workflow for {ticker}")
        
        workflow_results = {
            "ticker": ticker,
            "success": False,
            "steps": {},
            "summary": {
                "total_steps": 4,
                "completed_steps": 0,
                "failed_steps": 0,
                "files_processed": 0,
                "embeddings_generated": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 1: PDF Document Verification
        if self.config.enable_verification:
            verification_result = await self._execute_pdf_verification(ticker)
            workflow_results["steps"]["verification"] = verification_result
            
            if verification_result.success:
                workflow_results["summary"]["completed_steps"] += 1
            else:
                workflow_results["summary"]["failed_steps"] += 1
        
        # Step 2: PDF Download (if needed)
        needs_download = (
            self.config.enable_download and 
            workflow_results["steps"].get("verification", {}).get("data", {}).get("needs_download", False)
        )
        
        if needs_download:
            download_result = await self._execute_pdf_download(ticker)
            workflow_results["steps"]["download"] = download_result
            
            if download_result.success:
                workflow_results["summary"]["completed_steps"] += 1
                workflow_results["summary"]["files_processed"] += download_result.files_processed
            else:
                workflow_results["summary"]["failed_steps"] += 1
        
        # Step 3: Document Chunking
        if self.config.enable_chunking and self.pdf_processor:
            chunking_result = await self._execute_document_chunking(ticker)
            workflow_results["steps"]["chunking"] = chunking_result
            
            if chunking_result.success:
                workflow_results["summary"]["completed_steps"] += 1
                workflow_results["summary"]["files_processed"] += chunking_result.files_processed
            else:
                workflow_results["summary"]["failed_steps"] += 1
        
        # Step 4: Embedding Generation
        if (self.config.enable_embeddings and self.vector_store and 
            workflow_results["steps"].get("chunking", {}).get("success", False)):
            
            embedding_result = await self._execute_embedding_generation(
                ticker, workflow_results["steps"]["chunking"]
            )
            workflow_results["steps"]["embeddings"] = embedding_result
            
            if embedding_result.success:
                workflow_results["summary"]["completed_steps"] += 1
                workflow_results["summary"]["embeddings_generated"] = embedding_result.data.get("embeddings_count", 0)
            else:
                workflow_results["summary"]["failed_steps"] += 1
        
        # Determine overall success
        workflow_results["success"] = workflow_results["summary"]["completed_steps"] > 0
        
        logger.info(f"✅ PDF workflow completed for {ticker}: {workflow_results['summary']['completed_steps']}/{workflow_results['summary']['total_steps']} steps")
        return workflow_results
    
    async def _execute_pdf_verification(self, ticker: str) -> PDFWorkflowResult:
        """Execute PDF document verification step."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.hk_data_downloader:
                return PDFWorkflowResult(
                    step="verification",
                    success=False,
                    error="HK data downloader not available"
                )
            
            # Check for existing HKEX annual report PDFs
            try:
                verification_data = await self.hk_data_downloader.verify_pdf_documents(ticker)
            except AttributeError:
                # Method doesn't exist, return fallback
                verification_data = {
                    "status": "fallback",
                    "reason": "verify_pdf_documents method not available",
                    "needs_download": False,
                    "existing_files": []
                }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return PDFWorkflowResult(
                step="verification",
                success=True,
                data=verification_data,
                execution_time=execution_time,
                files_processed=len(verification_data.get("existing_files", []))
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.processing_errors += 1
            
            return PDFWorkflowResult(
                step="verification",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_pdf_download(self, ticker: str) -> PDFWorkflowResult:
        """Execute PDF download step."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.hk_data_downloader:
                return PDFWorkflowResult(
                    step="download",
                    success=False,
                    error="HK data downloader not available"
                )
            
            # Download latest HKEX annual reports
            try:
                download_data = await self.hk_data_downloader.download_hkex_pdfs(ticker)
            except AttributeError:
                # Method doesn't exist, return fallback
                download_data = {
                    "status": "fallback",
                    "reason": "download_hkex_pdfs method not available",
                    "downloaded_files": []
                }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            files_downloaded = len(download_data.get("downloaded_files", []))
            
            return PDFWorkflowResult(
                step="download",
                success=True,
                data=download_data,
                execution_time=execution_time,
                files_processed=files_downloaded
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.processing_errors += 1
            
            return PDFWorkflowResult(
                step="download",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_document_chunking(self, ticker: str) -> PDFWorkflowResult:
        """Execute document chunking step."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.pdf_processor:
                return PDFWorkflowResult(
                    step="chunking",
                    success=False,
                    error="PDF processor not available"
                )
            
            # Process downloaded PDFs into chunks with metadata
            try:
                chunking_data = await self.pdf_processor.process_ticker_documents(ticker)
            except AttributeError:
                # Method doesn't exist, return fallback
                chunking_data = {
                    "status": "fallback",
                    "reason": "process_ticker_documents method not available",
                    "chunks": [],
                    "processed_files": []
                }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            files_processed = len(chunking_data.get("processed_files", []))
            chunks_created = len(chunking_data.get("chunks", []))
            
            self.processed_documents += files_processed
            
            return PDFWorkflowResult(
                step="chunking",
                success=True,
                data=chunking_data,
                execution_time=execution_time,
                files_processed=files_processed
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.processing_errors += 1
            
            return PDFWorkflowResult(
                step="chunking",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _execute_embedding_generation(self, ticker: str, chunking_result: PDFWorkflowResult) -> PDFWorkflowResult:
        """Execute embedding generation step."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.vector_store:
                return PDFWorkflowResult(
                    step="embeddings",
                    success=False,
                    error="Vector store not available"
                )
            
            # Generate 384-dimensional embeddings using all-MiniLM-L6-v2
            try:
                embedding_data = await self.vector_store.generate_embeddings(ticker, chunking_result.data)
            except AttributeError:
                # Method doesn't exist, return fallback
                embedding_data = {
                    "status": "fallback",
                    "reason": "generate_embeddings method not available",
                    "embeddings_count": 0
                }
            
            execution_time = asyncio.get_event_loop().time() - start_time
            embeddings_count = embedding_data.get("embeddings_count", 0)
            
            self.generated_embeddings += embeddings_count
            
            return PDFWorkflowResult(
                step="embeddings",
                success=True,
                data=embedding_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.processing_errors += 1
            
            return PDFWorkflowResult(
                step="embeddings",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get PDF processing statistics."""
        return {
            "processed_documents": self.processed_documents,
            "generated_embeddings": self.generated_embeddings,
            "processing_errors": self.processing_errors,
            "components_available": {
                "hk_data_downloader": self.hk_data_downloader is not None,
                "pdf_processor": self.pdf_processor is not None,
                "vector_store": self.vector_store is not None
            },
            "config": {
                "verification_enabled": self.config.enable_verification,
                "download_enabled": self.config.enable_download,
                "chunking_enabled": self.config.enable_chunking,
                "embeddings_enabled": self.config.enable_embeddings,
                "chunk_size": self.config.chunk_size,
                "embedding_model": self.config.embedding_model
            }
        }
