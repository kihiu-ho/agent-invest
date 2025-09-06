"""
Streamlined PDF Document Processing Pipeline

This module orchestrates the complete streamlined pipeline:
PDF Download → Text Extraction (PyMuPDF only) → Embedding Generation → Vector Storage → Semantic Query

Features:
- End-to-end PDF processing without OCR fallback
- Automatic embedding generation and vector storage
- Natural language query interface
- Citation tracking and confidence scoring
- Integration with existing financial analysis system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from streamlined_pdf_processor import StreamlinedPDFProcessor
from streamlined_vector_store import StreamlinedVectorStore

logger = logging.getLogger(__name__)

class StreamlinedPipeline:
    """
    Complete streamlined pipeline for PDF document processing.
    
    Orchestrates: PDF Download → PyMuPDF Extraction → Embedding → Vector Storage → Query
    """
    
    def __init__(self, 
                 download_dir: str = "downloads/hkex_reports",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 weaviate_collection: str = "HKEXDocuments",
                 similarity_threshold: float = 0.7):
        """
        Initialize the streamlined pipeline.
        
        Args:
            download_dir: Directory for PDF downloads
            embedding_model: Sentence transformer model
            weaviate_collection: Weaviate collection name
            similarity_threshold: Minimum similarity for search results
        """
        self.download_dir = download_dir
        self.embedding_model = embedding_model
        self.weaviate_collection = weaviate_collection
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.pdf_processor = StreamlinedPDFProcessor(
            download_dir=download_dir,
            embedding_model=embedding_model
        )
        
        self.vector_store = StreamlinedVectorStore(
            collection_name=weaviate_collection,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"🚀 Streamlined Pipeline initialized")
        logger.info(f"   Download directory: {download_dir}")
        logger.info(f"   Embedding model: {embedding_model}")
        logger.info(f"   Vector collection: {weaviate_collection}")
    
    async def process_document(self, pdf_url: str, ticker: str, 
                             document_title: str = None, 
                             store_embeddings: bool = True) -> Dict[str, Any]:
        """
        Process a single PDF document through the complete pipeline.
        
        Args:
            pdf_url: URL to download PDF from
            ticker: Stock ticker (e.g., '0700.HK')
            document_title: Optional document title
            store_embeddings: Whether to store embeddings in vector database
            
        Returns:
            Dict with complete processing results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🚀 Starting streamlined processing for {ticker}")
            
            # Step 1: Process PDF (Download → Extract → Parse → Embed)
            processing_result = await self.pdf_processor.process_pdf_complete(
                pdf_url, ticker, document_title
            )
            
            if not processing_result.get("success"):
                return {
                    "success": False,
                    "error": "PDF processing failed",
                    "details": processing_result,
                    "ticker": ticker
                }
            
            # Step 2: Store embeddings in vector database (optional)
            storage_result = None
            if store_embeddings and processing_result.get("chunks"):
                logger.info(f"💾 Storing embeddings in vector database")
                
                # Connect to vector store
                connected = await self.vector_store.connect()
                if connected:
                    try:
                        storage_result = await self.vector_store.store_document_chunks(
                            processing_result["chunks"]
                        )
                    finally:
                        await self.vector_store.disconnect()
                else:
                    logger.warning("⚠️ Could not connect to vector store, skipping embedding storage")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "ticker": ticker,
                "pdf_url": pdf_url,
                "document_title": document_title,
                "processing": processing_result,
                "storage": storage_result,
                "pipeline_time": processing_time,
                "summary": {
                    "extraction_method": "pymupdf",
                    "total_pages": processing_result.get("total_pages", 0),
                    "quality_score": processing_result.get("quality_score", 0.0),
                    "sections_parsed": len(processing_result.get("sections", {})),
                    "chunks_created": processing_result.get("total_chunks", 0),
                    "embeddings_stored": storage_result.get("stored_count", 0) if storage_result else 0
                }
            }
            
            logger.info(f"✅ Streamlined processing completed for {ticker} in {processing_time:.2f}s")
            logger.info(f"   📊 {result['summary']['sections_parsed']} sections, {result['summary']['chunks_created']} chunks")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Streamlined processing failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
                "pipeline_time": processing_time
            }
    
    async def query_documents(self, query: str, ticker: str = None, 
                            section_filter: str = None) -> Dict[str, Any]:
        """
        Query processed documents using natural language.
        
        Args:
            query: Natural language query
            ticker: Optional ticker filter
            section_filter: Optional section filter
            
        Returns:
            Dict with query results and citations
        """
        try:
            logger.info(f"🔍 Querying documents: '{query[:50]}...'")
            
            # Connect to vector store
            connected = await self.vector_store.connect()
            if not connected:
                return {
                    "success": False,
                    "error": "Could not connect to vector store",
                    "query": query
                }
            
            try:
                # Perform query
                result = await self.vector_store.query_documents(
                    query, ticker, section_filter
                )
                
                logger.info(f"✅ Query completed: {result.get('total_results', 0)} results found")
                return result
                
            finally:
                await self.vector_store.disconnect()
                
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def process_multiple_documents(self, documents: List[Dict[str, str]], 
                                       store_embeddings: bool = True) -> Dict[str, Any]:
        """
        Process multiple PDF documents in batch.
        
        Args:
            documents: List of dicts with 'pdf_url', 'ticker', and optional 'document_title'
            store_embeddings: Whether to store embeddings in vector database
            
        Returns:
            Dict with batch processing results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"📚 Starting batch processing of {len(documents)} documents")
            
            results = []
            successful = 0
            failed = 0
            
            for i, doc in enumerate(documents, 1):
                logger.info(f"📄 Processing document {i}/{len(documents)}: {doc['ticker']}")
                
                result = await self.process_document(
                    doc['pdf_url'],
                    doc['ticker'],
                    doc.get('document_title'),
                    store_embeddings
                )
                
                results.append(result)
                
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1
                
                # Small delay between documents to be respectful
                if i < len(documents):
                    await asyncio.sleep(1)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            summary = {
                "total_documents": len(documents),
                "successful": successful,
                "failed": failed,
                "processing_time": processing_time,
                "average_time_per_doc": processing_time / len(documents) if documents else 0
            }
            
            logger.info(f"✅ Batch processing completed: {successful}/{len(documents)} successful")
            
            return {
                "success": True,
                "summary": summary,
                "results": results
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Batch processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and configuration.
        
        Returns:
            Dict with pipeline status information
        """
        return {
            "pipeline_type": "streamlined",
            "extraction_method": "pymupdf_only",
            "ocr_fallback": False,
            "embedding_model": self.embedding_model,
            "vector_collection": self.weaviate_collection,
            "similarity_threshold": self.similarity_threshold,
            "download_directory": self.download_dir,
            "components": {
                "pdf_processor": "StreamlinedPDFProcessor",
                "vector_store": "StreamlinedVectorStore",
                "embedding_model_loaded": self.pdf_processor.embedding_model is not None,
                "vector_store_available": self.vector_store.embedding_model is not None
            }
        }
