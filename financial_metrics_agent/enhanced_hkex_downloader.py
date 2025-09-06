#!/usr/bin/env python3
"""
Enhanced HKEX Annual Report Downloader

This module provides a comprehensive HKEX annual report downloader that integrates
with the existing AgentInvest infrastructure while adding enhanced features for:
- Specific ticker-based downloads (XXXX.HK format)
- Advanced PDF processing with intelligent chunking
- Comprehensive metadata extraction and storage
- Batch embedding generation and Weaviate storage
- Robust error handling and performance optimization

Features:
- Integration with existing HKEXDownloader and StreamlinedPDFProcessor
- Enhanced metadata structure with page numbers, sections, and document details
- Batch processing for optimal Weaviate performance
- Ticker-based filtering and semantic search capabilities
- Comprehensive logging and error recovery
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass, asdict

# Import existing AgentInvest components
try:
    from hkex_downloader import HKEXDownloader
    from streamlined_pdf_processor import StreamlinedPDFProcessor
    from weaviate_client import WeaviateClient
    from streamlined_vector_store import StreamlinedVectorStore
except ImportError as e:
    logging.warning(f"Some AgentInvest components not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Comprehensive metadata structure for HKEX documents."""
    ticker: str
    stock_code: str
    document_title: str
    report_year: Optional[int]
    document_type: str
    source_url: str
    download_date: str
    file_path: str
    file_size: int
    total_pages: int
    extraction_method: str
    processing_date: str
    confidence_score: float

@dataclass
class ChunkMetadata:
    """Metadata structure for document chunks."""
    chunk_id: str
    ticker: str
    stock_code: str
    document_title: str
    section_name: str
    page_number: int
    page_range: List[int]
    chunk_index: int
    char_count: int
    word_count: int
    content_type: str
    confidence_score: float
    extraction_method: str
    processed_date: str
    source_url: str

@dataclass
class ProcessingConfig:
    """Configuration for PDF processing and chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 50
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_section_detection: bool = True
    enable_page_tracking: bool = True

class EnhancedHKEXDownloader:
    """
    Enhanced HKEX Annual Report Downloader with comprehensive processing capabilities.
    
    This class integrates with existing AgentInvest infrastructure while providing
    enhanced features for document processing, metadata extraction, and vector storage.
    """
    
    def __init__(self, 
                 config: Optional[ProcessingConfig] = None,
                 weaviate_client: Optional[WeaviateClient] = None,
                 download_dir: Optional[str] = None):
        """
        Initialize the enhanced HKEX downloader.
        
        Args:
            config: Processing configuration
            weaviate_client: Weaviate client for vector storage
            download_dir: Directory for downloaded files
        """
        self.config = config or ProcessingConfig()
        self.download_dir = Path(download_dir or "./enhanced_hkex_downloads")
        self.download_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.hkex_downloader = HKEXDownloader()
        self.pdf_processor = StreamlinedPDFProcessor()
        self.weaviate_client = weaviate_client
        
        # Performance tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_generated": 0,
            "chunks_stored": 0,
            "processing_time": 0,
            "errors": []
        }
        
        logger.info(f"✅ Enhanced HKEX Downloader initialized with config: {asdict(self.config)}")

    async def download_and_process_ticker(self, 
                                        ticker: str,
                                        force_refresh: bool = False,
                                        max_reports: int = 1) -> Dict[str, Any]:
        """
        Download and process annual reports for a specific ticker.
        
        Args:
            ticker: Hong Kong stock ticker (e.g., '0005.HK')
            force_refresh: Whether to force re-download existing files
            max_reports: Maximum number of reports to process
            
        Returns:
            Dictionary containing complete processing results
        """
        start_time = time.time()
        logger.info(f"🚀 Starting enhanced processing for ticker: {ticker}")
        
        # Normalize ticker format
        normalized_ticker = self._normalize_ticker(ticker)
        stock_code = normalized_ticker.replace('.HK', '').zfill(4)
        
        result = {
            "ticker": normalized_ticker,
            "stock_code": stock_code,
            "success": False,
            "documents_processed": 0,
            "chunks_generated": 0,
            "chunks_stored": 0,
            "processing_time": 0,
            "metadata": {},
            "errors": []
        }
        
        try:
            # Step 1: Download annual reports
            logger.info(f"📥 Step 1: Downloading reports for {normalized_ticker}")
            download_result = await self._download_reports(normalized_ticker, force_refresh, max_reports)
            
            if not download_result.get("success"):
                result["errors"].append(f"Download failed: {download_result.get('error', 'Unknown error')}")
                return result
            
            downloaded_files = download_result.get("files", [])
            logger.info(f"✅ Downloaded {len(downloaded_files)} files")
            
            # Step 2: Process each downloaded file
            all_chunks = []
            document_metadata = []
            
            for file_info in downloaded_files:
                file_path = file_info.get("file_path")
                if not file_path or not Path(file_path).exists():
                    continue
                    
                logger.info(f"📄 Processing file: {Path(file_path).name}")
                
                # Process PDF and generate chunks
                processing_result = await self._process_pdf_file(
                    file_path, normalized_ticker, file_info
                )
                
                if processing_result.get("success"):
                    all_chunks.extend(processing_result.get("chunks", []))
                    document_metadata.append(processing_result.get("metadata"))
                    result["documents_processed"] += 1
                else:
                    result["errors"].append(f"Processing failed for {Path(file_path).name}")
            
            result["chunks_generated"] = len(all_chunks)
            logger.info(f"📊 Generated {len(all_chunks)} chunks from {result['documents_processed']} documents")
            
            # Step 3: Store chunks in Weaviate
            if all_chunks and self.weaviate_client:
                logger.info(f"💾 Step 3: Storing {len(all_chunks)} chunks in Weaviate")
                storage_result = await self._store_chunks_in_weaviate(all_chunks, normalized_ticker)
                result["chunks_stored"] = storage_result.get("chunks_stored", 0)
                
                if not storage_result.get("success"):
                    result["errors"].extend(storage_result.get("errors", []))
            
            # Update statistics
            result["processing_time"] = time.time() - start_time
            result["metadata"] = {
                "documents": document_metadata,
                "config": asdict(self.config),
                "processing_date": datetime.now().isoformat()
            }
            result["success"] = result["documents_processed"] > 0
            
            # Update global stats
            self.stats["documents_processed"] += result["documents_processed"]
            self.stats["chunks_generated"] += result["chunks_generated"]
            self.stats["chunks_stored"] += result["chunks_stored"]
            self.stats["processing_time"] += result["processing_time"]
            
            logger.info(f"✅ Enhanced processing completed for {normalized_ticker}")
            logger.info(f"📊 Results: {result['documents_processed']} docs, {result['chunks_generated']} chunks, {result['chunks_stored']} stored")
            
            return result
            
        except Exception as e:
            result["errors"].append(f"Unexpected error: {str(e)}")
            result["processing_time"] = time.time() - start_time
            logger.error(f"❌ Enhanced processing failed for {ticker}: {e}")
            return result

    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker to XXXX.HK format."""
        if not ticker:
            raise ValueError("Ticker cannot be empty")
        
        # Remove any whitespace
        ticker = ticker.strip().upper()
        
        # If already in correct format, return as-is
        if ticker.endswith('.HK') and len(ticker.split('.')[0]) <= 4:
            return ticker
        
        # If just the stock code, add .HK suffix
        if ticker.isdigit():
            return f"{ticker.zfill(4)}.HK"
        
        # If has .HK but stock code needs padding
        if '.HK' in ticker:
            stock_code = ticker.replace('.HK', '')
            return f"{stock_code.zfill(4)}.HK"
        
        # Default: assume it's a stock code and add .HK
        return f"{ticker.zfill(4)}.HK"

    async def _download_reports(self, 
                              ticker: str, 
                              force_refresh: bool, 
                              max_reports: int) -> Dict[str, Any]:
        """Download annual reports using existing infrastructure."""
        try:
            # Use existing HKEXDownloader
            download_result = await self.hkex_downloader.process_and_store_document(ticker)
            
            if download_result.get("success"):
                # Extract file information from the result
                files = []
                download_info = download_result.get("download_result", {})
                
                # Handle different result formats from existing downloaders
                if "filepath" in download_info:
                    files.append({
                        "file_path": download_info["filepath"],
                        "url": download_info.get("url", ""),
                        "title": download_info.get("title", f"{ticker} Annual Report"),
                        "year": download_info.get("year"),
                        "file_size": download_info.get("file_size", 0)
                    })
                
                return {
                    "success": True,
                    "files": files,
                    "download_method": "hkex_downloader"
                }
            else:
                return {
                    "success": False,
                    "error": download_result.get("message", "Download failed"),
                    "files": []
                }
                
        except Exception as e:
            logger.error(f"❌ Download failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "files": []
            }

    async def _process_pdf_file(self,
                              file_path: str,
                              ticker: str,
                              file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single PDF file with enhanced chunking and metadata extraction.

        Args:
            file_path: Path to the PDF file
            ticker: Stock ticker
            file_info: File information from download

        Returns:
            Dictionary containing processing results and chunks
        """
        try:
            logger.info(f"📄 Processing PDF: {Path(file_path).name}")

            # Extract text using existing PDF processor
            extraction_result = await self.pdf_processor.extract_text_enhanced(file_path, ticker)

            if not extraction_result.get("success"):
                return {
                    "success": False,
                    "error": f"Text extraction failed: {extraction_result.get('error')}",
                    "chunks": [],
                    "metadata": None
                }

            # Get page texts and raw text
            page_texts = extraction_result.get("page_texts", [])
            raw_text = extraction_result.get("raw_text", "")

            # Create document metadata with enhanced source URL handling
            source_url = file_info.get("url", "")

            # Enhanced URL preservation logic with filename construction
            if not source_url or source_url in ["local_file", "unknown", ""]:
                # Try alternative URL fields from download metadata
                source_url = (
                    file_info.get("source_url") or
                    file_info.get("download_url") or
                    file_info.get("pdf_url") or
                    file_info.get("original_url") or
                    ""
                )

            # If still no URL, try to construct from filename and metadata
            if not source_url or source_url in ["local_file", "unknown", ""]:
                # Extract filename from file_path
                filename = Path(file_path).name if file_path else ""

                # Try to construct complete URL with filename
                if filename and filename.endswith('.pdf'):
                    # Check if we have year information to construct proper HKEX URL
                    year = file_info.get("year") or self._extract_year_from_filename(filename)
                    stock_code = ticker.replace('.HK', '').zfill(4)

                    if year and year != "unknown":
                        # Construct HKEX-style URL with year and filename
                        source_url = f"https://www.hkexnews.hk/listedco/listconews/sehk/{year}/{year[:4]}/{filename}"
                    else:
                        # Fallback to filename-based URL
                        source_url = f"https://www.hkexnews.hk/listedco/{stock_code}/{filename}"
                else:
                    # Final fallback to directory URL
                    source_url = f"https://www.hkexnews.hk/listedco/{ticker.replace('.HK', '').zfill(4)}/"

            # Ensure proper HTTP URL format
            if not source_url.startswith("http"):
                if source_url.startswith("/"):
                    source_url = f"https://www.hkexnews.hk{source_url}"
                else:
                    # If it's just a filename, construct full URL
                    if source_url.endswith('.pdf'):
                        stock_code = ticker.replace('.HK', '').zfill(4)
                        source_url = f"https://www.hkexnews.hk/listedco/{stock_code}/{source_url}"
                    else:
                        source_url = f"https://www.hkexnews.hk/listedco/{ticker.replace('.HK', '').zfill(4)}/"

            # Enhanced document title construction
            document_title = self._construct_enhanced_document_title(
                file_info, ticker, file_path, raw_text
            )

            doc_metadata = DocumentMetadata(
                ticker=ticker,
                stock_code=ticker.replace('.HK', '').zfill(4),
                document_title=document_title,
                report_year=file_info.get("year"),
                document_type="annual_report",
                source_url=source_url,  # Use the enhanced source URL
                download_date=datetime.now().isoformat(),
                file_path=file_path,
                file_size=file_info.get("file_size", Path(file_path).stat().st_size),
                total_pages=len(page_texts),
                extraction_method="pymupdf",
                processing_date=datetime.now().isoformat(),
                confidence_score=extraction_result.get("quality_score", 0.8)
            )

            # Generate chunks with enhanced metadata
            all_chunks = []

            # Process page-by-page for better metadata tracking
            for page_data in page_texts:
                page_number = page_data.get("page_number", 0)
                page_text = page_data.get("text", "")

                if len(page_text.strip()) < self.config.min_chunk_size:
                    continue

                # Create chunks for this page
                page_chunks = self._create_enhanced_chunks(
                    page_text, page_number, doc_metadata
                )
                all_chunks.extend(page_chunks)

            # Also try to detect sections for better organization
            if self.config.enable_section_detection:
                section_chunks = await self._create_section_based_chunks(
                    raw_text, page_texts, doc_metadata
                )
                all_chunks.extend(section_chunks)

            logger.info(f"✅ Generated {len(all_chunks)} chunks from {len(page_texts)} pages")

            return {
                "success": True,
                "chunks": all_chunks,
                "metadata": asdict(doc_metadata),
                "pages_processed": len(page_texts),
                "chunks_generated": len(all_chunks)
            }

        except Exception as e:
            logger.error(f"❌ PDF processing failed for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks": [],
                "metadata": None
            }

    def _create_enhanced_chunks(self,
                              text: str,
                              page_number: int,
                              doc_metadata: DocumentMetadata) -> List[Dict[str, Any]]:
        """
        Create enhanced chunks with comprehensive metadata.

        Args:
            text: Text to chunk
            page_number: Page number for the text
            doc_metadata: Document metadata

        Returns:
            List of chunks with enhanced metadata
        """
        chunks = []

        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.config.chunk_size, len(text))

            # Try to break at word boundaries
            if end < len(text):
                # Look for the last space within reasonable distance
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.config.chunk_size * 0.8:
                    end = last_space

            chunk_text = text[start:end].strip()

            # Skip very small chunks
            if len(chunk_text) < self.config.min_chunk_size:
                start = end
                continue

            # Create chunk metadata
            chunk_id = self._generate_chunk_id(doc_metadata.ticker, page_number, chunk_index)

            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                ticker=doc_metadata.ticker,
                stock_code=doc_metadata.stock_code,
                document_title=doc_metadata.document_title,
                section_name=f"Page {page_number}",
                page_number=page_number,
                page_range=[page_number],
                chunk_index=chunk_index,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                content_type="page_content",
                confidence_score=doc_metadata.confidence_score,
                extraction_method=doc_metadata.extraction_method,
                processed_date=doc_metadata.processing_date,
                source_url=doc_metadata.source_url
            )

            # Create chunk object
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": asdict(chunk_metadata),
                "embedding": None  # Will be generated during storage
            }

            chunks.append(chunk)
            chunk_index += 1

            # Move start position with overlap
            start = max(start + self.config.chunk_size - self.config.chunk_overlap, end)

        return chunks

    async def _create_section_based_chunks(self,
                                         raw_text: str,
                                         page_texts: List[Dict],
                                         doc_metadata: DocumentMetadata) -> List[Dict[str, Any]]:
        """
        Create chunks based on document sections for better semantic organization.

        Args:
            raw_text: Complete document text
            page_texts: Page-by-page text data
            doc_metadata: Document metadata

        Returns:
            List of section-based chunks
        """
        try:
            # Use existing section detection from StreamlinedPDFProcessor
            sections = self.pdf_processor.extract_sections_from_pages(page_texts)

            section_chunks = []

            for section_name, section_data in sections.items():
                section_text = section_data.get("content", "")
                section_pages = section_data.get("pages", [])

                if len(section_text.strip()) < self.config.min_chunk_size:
                    continue

                # Create chunks for this section
                start = 0
                chunk_index = 0

                while start < len(section_text):
                    end = min(start + self.config.chunk_size, len(section_text))

                    # Try to break at sentence boundaries for sections
                    if end < len(section_text):
                        last_period = section_text.rfind('.', start, end)
                        if last_period > start + self.config.chunk_size * 0.7:
                            end = last_period + 1

                    chunk_text = section_text[start:end].strip()

                    if len(chunk_text) < self.config.min_chunk_size:
                        start = end
                        continue

                    # Create section chunk metadata
                    chunk_id = self._generate_chunk_id(
                        doc_metadata.ticker,
                        f"section_{section_name}",
                        chunk_index
                    )

                    chunk_metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        ticker=doc_metadata.ticker,
                        stock_code=doc_metadata.stock_code,
                        document_title=doc_metadata.document_title,
                        section_name=section_name,
                        page_number=section_pages[0] if section_pages else 0,
                        page_range=section_pages,
                        chunk_index=chunk_index,
                        char_count=len(chunk_text),
                        word_count=len(chunk_text.split()),
                        content_type="section_content",
                        confidence_score=section_data.get("confidence", 0.8),
                        extraction_method=doc_metadata.extraction_method,
                        processed_date=doc_metadata.processing_date,
                        source_url=doc_metadata.source_url
                    )

                    chunk = {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": asdict(chunk_metadata),
                        "embedding": None
                    }

                    section_chunks.append(chunk)
                    chunk_index += 1

                    start = max(start + self.config.chunk_size - self.config.chunk_overlap, end)

            logger.info(f"✅ Generated {len(section_chunks)} section-based chunks")
            return section_chunks

        except Exception as e:
            logger.warning(f"⚠️ Section-based chunking failed: {e}")
            return []

    def _generate_chunk_id(self, ticker: str, page_or_section: Any, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        base_string = f"{ticker}_{page_or_section}_{chunk_index}_{datetime.now().isoformat()}"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]

    async def _store_chunks_in_weaviate(self,
                                      chunks: List[Dict[str, Any]],
                                      ticker: str) -> Dict[str, Any]:
        """
        Store chunks in Weaviate with batch processing for optimal performance.

        Args:
            chunks: List of chunks to store
            ticker: Stock ticker for logging

        Returns:
            Dictionary containing storage results
        """
        if not self.weaviate_client:
            logger.warning("⚠️ No Weaviate client available for storage")
            return {
                "success": False,
                "error": "No Weaviate client available",
                "chunks_stored": 0,
                "errors": []
            }

        logger.info(f"💾 Storing {len(chunks)} chunks for {ticker} in Weaviate")

        try:
            # Connect to Weaviate
            async with self.weaviate_client as client:
                # Process chunks in batches
                total_stored = 0
                errors = []

                for i in range(0, len(chunks), self.config.batch_size):
                    batch = chunks[i:i + self.config.batch_size]
                    batch_num = i // self.config.batch_size + 1
                    total_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size

                    logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

                    # Generate embeddings for batch
                    batch_with_embeddings = await self._generate_embeddings_for_batch(batch)

                    # Store batch in Weaviate
                    batch_result = await self._store_batch_in_weaviate(client, batch_with_embeddings)

                    if batch_result.get("success"):
                        total_stored += batch_result.get("stored_count", 0)
                        logger.info(f"✅ Batch {batch_num} stored successfully ({batch_result.get('stored_count', 0)} chunks)")
                    else:
                        error_msg = f"Batch {batch_num} failed: {batch_result.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        logger.error(f"❌ {error_msg}")

                success = total_stored > 0
                logger.info(f"📊 Storage complete: {total_stored}/{len(chunks)} chunks stored")

                return {
                    "success": success,
                    "chunks_stored": total_stored,
                    "total_chunks": len(chunks),
                    "batches_processed": total_batches,
                    "errors": errors
                }

        except Exception as e:
            logger.error(f"❌ Weaviate storage failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_stored": 0,
                "errors": [str(e)]
            }

    async def _generate_embeddings_for_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a batch of chunks.

        Args:
            batch: List of chunks without embeddings

        Returns:
            List of chunks with embeddings
        """
        try:
            # Use existing embedding generation from StreamlinedVectorStore
            if hasattr(self, 'vector_store') and self.vector_store:
                # Use existing vector store for embedding generation
                for chunk in batch:
                    if chunk.get("embedding") is None:
                        embedding = await self.vector_store._generate_embedding(chunk["text"])
                        chunk["embedding"] = embedding
            else:
                # Fallback: use sentence transformers directly
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(self.config.embedding_model)

                    texts = [chunk["text"] for chunk in batch]
                    embeddings = model.encode(texts, convert_to_tensor=False)

                    for chunk, embedding in zip(batch, embeddings):
                        chunk["embedding"] = embedding.tolist()

                except ImportError:
                    logger.warning("⚠️ SentenceTransformers not available, skipping embedding generation")
                    for chunk in batch:
                        chunk["embedding"] = None

            return batch

        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            # Return chunks without embeddings
            for chunk in batch:
                chunk["embedding"] = None
            return batch

    async def _store_batch_in_weaviate(self,
                                     client: WeaviateClient,
                                     batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store a single batch of chunks in Weaviate.

        Args:
            client: Weaviate client
            batch: Batch of chunks with embeddings

        Returns:
            Dictionary containing batch storage results
        """
        try:
            # Prepare objects for Weaviate storage
            weaviate_objects = []

            for chunk in batch:
                metadata = chunk.get("metadata", {})

                # Create Weaviate object following existing schema
                obj = {
                    "content": chunk["text"],
                    "ticker": metadata.get("ticker", ""),
                    "stock_code": metadata.get("stock_code", ""),
                    "document_title": metadata.get("document_title", ""),
                    "section_title": metadata.get("section_name", ""),
                    "page_number": metadata.get("page_number", 0),
                    "page_range": metadata.get("page_range", []),
                    "chunk_id": chunk["chunk_id"],
                    "content_type": metadata.get("content_type", ""),
                    "confidence_score": metadata.get("confidence_score", 0.8),
                    "extraction_method": metadata.get("extraction_method", ""),
                    "processed_date": metadata.get("processed_date", ""),
                    "source_url": metadata.get("source_url", ""),
                    "char_count": metadata.get("char_count", 0),
                    "word_count": metadata.get("word_count", 0),
                    "chunk_index": metadata.get("chunk_index", 0)
                }

                # Add vector if available
                if chunk.get("embedding"):
                    obj["vector"] = chunk["embedding"]

                weaviate_objects.append(obj)

            # Use existing client's batch storage method
            storage_result = await client.store_document_sections(
                ticker=batch[0]["metadata"]["ticker"],
                sections={f"chunk_{i}": obj for i, obj in enumerate(weaviate_objects)}
            )

            if storage_result.get("success"):
                return {
                    "success": True,
                    "stored_count": len(weaviate_objects),
                    "method": "weaviate_client"
                }
            else:
                return {
                    "success": False,
                    "error": storage_result.get("error", "Storage failed"),
                    "stored_count": 0
                }

        except Exception as e:
            logger.error(f"❌ Batch storage failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_count": 0
            }

    async def query_documents(self,
                            ticker: str,
                            query: str,
                            content_types: Optional[List[str]] = None,
                            limit: int = 10) -> Dict[str, Any]:
        """
        Query stored documents with ticker-based filtering.

        Args:
            ticker: Stock ticker to filter by
            query: Search query
            content_types: Optional content type filters
            limit: Maximum number of results

        Returns:
            Dictionary containing query results
        """
        if not self.weaviate_client:
            return {
                "success": False,
                "error": "No Weaviate client available",
                "results": []
            }

        try:
            normalized_ticker = self._normalize_ticker(ticker)
            logger.info(f"🔍 Querying documents for {normalized_ticker}: '{query}'")

            async with self.weaviate_client as client:
                # Use existing search method
                search_result = await client.search_documents(
                    ticker=normalized_ticker,
                    query=query,
                    content_types=content_types,
                    limit=limit
                )

                if isinstance(search_result, dict) and search_result.get("success"):
                    results = search_result.get("results", [])
                    logger.info(f"✅ Found {len(results)} relevant documents")

                    return {
                        "success": True,
                        "results": results,
                        "query": query,
                        "ticker": normalized_ticker,
                        "total_found": len(results)
                    }
                else:
                    return {
                        "success": False,
                        "error": search_result.get("error", "Search failed"),
                        "results": []
                    }

        except Exception as e:
            logger.error(f"❌ Document query failed for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "stats": self.stats.copy(),
            "config": asdict(self.config),
            "last_updated": datetime.now().isoformat()
        }

    def _extract_year_from_filename(self, filename: str) -> str:
        """
        Extract year from filename using various patterns.

        Args:
            filename: PDF filename to extract year from

        Returns:
            Year string or "unknown" if not found
        """
        import re

        # Common year patterns in HKEX filenames
        year_patterns = [
            r'(\d{4})',  # Any 4-digit year
            r'20(\d{2})',  # 20XX format
            r'_(\d{4})_',  # Year surrounded by underscores
            r'(\d{4})\.pdf'  # Year before .pdf extension
        ]

        for pattern in year_patterns:
            match = re.search(pattern, filename)
            if match:
                year = match.group(1)
                # Validate year is reasonable (2000-2030)
                if year.isdigit() and 2000 <= int(year) <= 2030:
                    return year

        return "unknown"

    def _construct_enhanced_document_title(
        self,
        file_info: Dict[str, Any],
        ticker: str,
        file_path: str,
        raw_text: str = ""
    ) -> str:
        """
        Construct enhanced document title with company name, document type, and year.

        Args:
            file_info: File information from download process
            ticker: Stock ticker (e.g., "0005.HK")
            file_path: Path to the PDF file
            raw_text: Extracted text content for title detection

        Returns:
            Enhanced document title
        """
        import re

        # Start with basic info
        stock_code = ticker.replace('.HK', '').zfill(4)
        year = file_info.get("year", self._extract_year_from_filename(Path(file_path).name))

        # Try to get company name from various sources
        company_name = self._extract_company_name(file_info, ticker, raw_text)

        # Try to get document type from title or filename
        doc_type = self._extract_document_type(file_info, Path(file_path).name, raw_text)

        # Construct enhanced title
        title_parts = []

        if company_name and company_name != "Unknown Company":
            title_parts.append(company_name)
        else:
            title_parts.append(f"Stock {stock_code}")

        title_parts.append(doc_type)

        if year and year != "unknown":
            title_parts.append(year)

        return " ".join(title_parts)

    def _extract_company_name(self, file_info: Dict[str, Any], ticker: str, raw_text: str) -> str:
        """Extract company name from various sources."""
        import re

        # Try from file_info first
        if "company_name" in file_info:
            return file_info["company_name"]

        # Try from title
        title = file_info.get("title", "")
        if title and len(title) > 5:
            # Clean up title to extract company name
            clean_title = re.sub(r'(annual report|interim report|financial statements)', '', title.lower())
            clean_title = re.sub(r'\d{4}', '', clean_title)  # Remove years
            clean_title = clean_title.strip()
            if len(clean_title) > 3:
                return clean_title.title()

        # Try from raw text (first few lines often contain company name)
        if raw_text:
            lines = raw_text.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                line = line.strip()
                if len(line) > 10 and len(line) < 100:
                    # Look for company name patterns
                    if any(keyword in line.lower() for keyword in ['limited', 'ltd', 'holdings', 'group', 'company']):
                        # Clean up the line
                        clean_line = re.sub(r'[^\w\s&\-\(\)]', ' ', line)
                        clean_line = ' '.join(clean_line.split())
                        if len(clean_line) > 5:
                            return clean_line

        # Fallback: Use stock code mapping if available
        stock_code = ticker.replace('.HK', '').zfill(4)
        company_mappings = {
            "0005": "HSBC Holdings",
            "0700": "Tencent Holdings",
            "0941": "China Mobile",
            "1299": "AIA Group",
            "2318": "Ping An Insurance",
            "3988": "Bank of China"
        }

        return company_mappings.get(stock_code, "Unknown Company")

    def _extract_document_type(self, file_info: Dict[str, Any], filename: str, raw_text: str) -> str:
        """Extract document type from various sources."""
        import re

        # Check title first
        title = file_info.get("title", "").lower()
        filename_lower = filename.lower()

        # Document type patterns
        if any(keyword in title or keyword in filename_lower for keyword in ['interim', 'half', 'h1', 'h2']):
            return "Interim Report"
        elif any(keyword in title or keyword in filename_lower for keyword in ['annual', 'yearly']):
            return "Annual Report"
        elif any(keyword in title or keyword in filename_lower for keyword in ['quarterly', 'q1', 'q2', 'q3', 'q4']):
            return "Quarterly Report"
        elif any(keyword in title or keyword in filename_lower for keyword in ['financial statements', 'financials']):
            return "Financial Statements"

        # Check raw text for document type indicators
        if raw_text:
            text_sample = raw_text[:2000].lower()  # Check first 2000 characters
            if 'interim report' in text_sample:
                return "Interim Report"
            elif 'annual report' in text_sample:
                return "Annual Report"
            elif 'quarterly report' in text_sample:
                return "Quarterly Report"

        # Default fallback
        return "Annual Report"
