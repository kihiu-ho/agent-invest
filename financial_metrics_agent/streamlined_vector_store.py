"""
Streamlined Vector Store for PDF Document Embeddings

This module provides vector storage and semantic search capabilities
for the streamlined PDF processing pipeline.

Features:
- Weaviate vector database integration
- Batch embedding storage with metadata
- Semantic search with similarity scoring
- Natural language query processing
- Citation and confidence tracking
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from weaviate.classes.config import Configure, Property, DataType
    import weaviate.classes.config
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StreamlinedVectorStore:
    """
    Streamlined vector store for PDF document embeddings.
    
    Provides storage and semantic search capabilities for processed PDF content.
    """
    
    def __init__(self, 
                 weaviate_url: str = None,
                 weaviate_api_key: str = None,
                 collection_name: str = "HKEXDocuments",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7):
        """
        Initialize the vector store.
        
        Args:
            weaviate_url: Weaviate cluster URL
            weaviate_api_key: Weaviate API key
            collection_name: Name of the Weaviate collection
            embedding_model: Sentence transformer model name
            similarity_threshold: Minimum similarity score for results
        """
        self.weaviate_url = weaviate_url or os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = weaviate_api_key or os.getenv("WEAVIATE_API_KEY")
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # Load embedding model
        self._load_embedding_model()
        
        logger.info(f"🗄️ Streamlined Vector Store initialized")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Embedding model: {embedding_model}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for query embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✅ Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def connect(self) -> bool:
        """
        Connect to Weaviate cluster and ensure collection exists.

        Returns:
            True if connection successful, False otherwise
        """
        if not WEAVIATE_AVAILABLE:
            logger.error("❌ Weaviate client not available")
            return False

        try:
            # Connect to Weaviate
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.weaviate_url,
                auth_credentials=weaviate.auth.AuthApiKey(self.weaviate_api_key)
            )

            # Create collection if it doesn't exist
            await self._create_collection_if_not_exists()

            # Get collection using the working pattern
            try:
                self.collection = self.client.collections.get(self.collection_name)

                # Test collection access with a simple operation
                if self.collection:
                    try:
                        # Test collection access - this will fail if collection is not accessible
                        stats = self.collection.aggregate.over_all(total_count=True)
                        logger.info(f"✅ Connected to Weaviate collection '{self.collection_name}' ({stats.total_count} objects)")
                        return True
                    except Exception as test_error:
                        logger.error(f"❌ Collection access test failed: {test_error}")
                        return False
                else:
                    logger.error(f"❌ Could not get collection '{self.collection_name}'")
                    return False

            except Exception as e:
                logger.error(f"❌ Could not access collection '{self.collection_name}': {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to connect to Weaviate: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Weaviate cluster."""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.collection = None
                logger.info("🔌 Disconnected from Weaviate")
        except Exception as e:
            logger.warning(f"⚠️ Error disconnecting from Weaviate: {e}")
            self.client = None
            self.collection = None
    
    async def store_document_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store document chunks with embeddings in Weaviate.
        
        Args:
            chunks: List of chunks with embeddings and metadata
            
        Returns:
            Dict with storage results
        """
        if not self.collection:
            logger.error("❌ Not connected to Weaviate")
            return {"success": False, "error": "Not connected to Weaviate"}
        
        try:
            stored_count = 0
            failed_count = 0
            
            # Store chunks in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Prepare batch data
                    batch_objects = []
                    for chunk in batch:
                        if "embedding" not in chunk:
                            logger.warning(f"⚠️ Chunk missing embedding, skipping")
                            failed_count += 1
                            continue
                        
                        # Prepare object for Weaviate with enhanced metadata
                        obj = {
                            "content": chunk["text"],
                            "ticker": chunk["metadata"]["ticker"],
                            "document_title": chunk["metadata"]["document_title"],
                            "source_url": chunk["metadata"]["source_url"],
                            "extraction_method": chunk["metadata"]["extraction_method"],
                            "processed_date": chunk["metadata"]["processed_date"],
                            "chunk_id": chunk["chunk_id"],
                            "char_count": chunk["char_count"],
                            "content_type": chunk["metadata"].get("content_type", "unknown")
                        }

                        # Add page-specific or section-specific metadata
                        if chunk["metadata"].get("content_type") == "page_content":
                            obj["page_number"] = chunk["metadata"].get("page_number", 0)
                            obj["section_title"] = "page_content"
                            obj["pages"] = [chunk["metadata"].get("page_number", 0)]
                            obj["confidence_score"] = 1.0  # Page content has high confidence
                        elif chunk["metadata"].get("content_type") == "section_content":
                            obj["section_title"] = chunk["metadata"].get("section_title", "unknown")
                            obj["pages"] = chunk["metadata"].get("pages", [])
                            obj["confidence_score"] = chunk["metadata"].get("confidence_score", 0.5)
                            obj["page_number"] = chunk["metadata"].get("pages", [0])[0] if chunk["metadata"].get("pages") else 0
                        else:
                            # Fallback for backward compatibility
                            obj["section_title"] = chunk["metadata"].get("section_title", "unknown")
                            obj["pages"] = chunk["metadata"].get("pages", [])
                            obj["confidence_score"] = chunk["metadata"].get("confidence_score", 0.5)
                            obj["page_number"] = 0
                        
                        batch_objects.append({
                            "properties": obj,
                            "vector": chunk["embedding"]
                        })
                    
                    # Use proper Weaviate v4 batch insert pattern
                    if batch_objects:
                        try:
                            with self.collection.batch.dynamic() as batch:
                                for obj_data in batch_objects:
                                    # Generate a unique UUID for each object
                                    import uuid
                                    object_uuid = str(uuid.uuid4())

                                    batch.add_object(
                                        properties=obj_data["properties"],
                                        vector=obj_data["vector"],
                                        uuid=object_uuid
                                    )

                            # Count successful inserts in this batch
                            batch_stored = len(batch_objects)
                            stored_count += batch_stored

                            if stored_count % 100 == 0:  # Log every 100 successful inserts
                                logger.info(f"✅ Stored {stored_count} chunks so far...")

                        except Exception as batch_error:
                            logger.warning(f"⚠️ Batch insert failed, falling back to individual inserts: {batch_error}")
                            # Fallback to individual inserts if batch fails
                            for obj_data in batch_objects:
                                try:
                                    result = self.collection.data.insert(
                                        properties=obj_data["properties"],
                                        vector=obj_data["vector"]
                                    )
                                    stored_count += 1
                                except Exception as insert_error:
                                    logger.warning(f"⚠️ Failed to insert individual chunk: {insert_error}")
                                    failed_count += 1
                
                except Exception as e:
                    logger.error(f"❌ Failed to store batch {i//batch_size + 1}: {e}")
                    failed_count += len(batch)
            
            logger.info(f"✅ Stored {stored_count} chunks, {failed_count} failed")
            
            return {
                "success": True,
                "stored_count": stored_count,
                "failed_count": failed_count,
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing document chunks: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def semantic_search(self, query: str, ticker: str = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on stored documents.
        
        Args:
            query: Natural language query
            ticker: Optional ticker filter
            limit: Maximum number of results
            
        Returns:
            List of search results with similarity scores
        """
        if not self.collection or not self.embedding_model:
            logger.error("❌ Vector store not properly initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Perform vector search with proper Weaviate v4 API syntax
            if ticker:
                # Try multiple approaches for filtered search
                try:
                    # Approach 1: Use Filter class with where parameter in near_vector
                    from weaviate.classes.query import Filter
                    response = self.collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=limit,
                        return_metadata=MetadataQuery(score=True, creation_time=True),
                        where=Filter.by_property("ticker").equal(ticker)
                    )
                except Exception as filter_error:
                    logger.warning(f"⚠️ Filtered search failed, using fallback: {filter_error}")
                    # Fallback: Get more results and filter manually
                    response = self.collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=limit * 5,  # Get more results to filter manually
                        return_metadata=MetadataQuery(score=True, creation_time=True)
                    )
            else:
                # No filter, just vector search
                response = self.collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit,
                    return_metadata=MetadataQuery(score=True, creation_time=True)
                )
            
            # Process results with manual filtering if needed
            results = []
            processed_count = 0

            for obj in response.objects:
                # Apply manual ticker filtering if needed (fallback case)
                if ticker and obj.properties.get("ticker", "") != ticker:
                    continue

                if obj.metadata.score >= self.similarity_threshold:
                    result = {
                        "content": obj.properties.get("content", ""),
                        "ticker": obj.properties.get("ticker", ""),
                        "document_title": obj.properties.get("document_title", ""),
                        "section_title": obj.properties.get("section_title", ""),
                        "source_url": obj.properties.get("source_url", ""),
                        "page_number": obj.properties.get("page_number", 0),
                        "pages": obj.properties.get("pages", []),
                        "confidence_score": obj.properties.get("confidence_score", 0.0),
                        "content_type": obj.properties.get("content_type", "unknown"),
                        "similarity_score": obj.metadata.score,
                        "chunk_id": obj.properties.get("chunk_id", 0),
                        "char_count": obj.properties.get("char_count", 0)
                    }
                    results.append(result)
                    processed_count += 1

                    # Limit results to requested amount
                    if processed_count >= limit:
                        break
            
            logger.info(f"🔍 Found {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in semantic search: {e}")
            return []

    async def query_documents(self, query: str, ticker: str = None,
                            section_filter: str = None) -> Dict[str, Any]:
        """
        Complete query pipeline with structured results and citations.

        Args:
            query: Natural language query
            ticker: Optional ticker filter
            section_filter: Optional section filter (e.g., 'financial_highlights')

        Returns:
            Dict with structured query results
        """
        start_time = datetime.now()

        try:
            # Perform semantic search
            search_results = await self.semantic_search(query, ticker, limit=20)

            # Filter by section if specified
            if section_filter:
                search_results = [
                    result for result in search_results
                    if result["section_title"] == section_filter
                ]

            # Group results by document and section
            grouped_results = {}
            for result in search_results:
                doc_key = f"{result['ticker']}_{result['document_title']}"
                section_key = result['section_title']

                if doc_key not in grouped_results:
                    grouped_results[doc_key] = {}

                if section_key not in grouped_results[doc_key]:
                    grouped_results[doc_key][section_key] = []

                grouped_results[doc_key][section_key].append(result)

            # Calculate aggregate scores and prepare citations
            citations = []
            citation_id = 1

            for doc_key, sections in grouped_results.items():
                for section_key, chunks in sections.items():
                    # Calculate average similarity score for this section
                    avg_similarity = sum(chunk["similarity_score"] for chunk in chunks) / len(chunks)

                    # Get representative chunk (highest similarity)
                    best_chunk = max(chunks, key=lambda x: x["similarity_score"])

                    citation = {
                        "citation_id": citation_id,
                        "ticker": best_chunk["ticker"],
                        "document_title": best_chunk["document_title"],
                        "section_title": best_chunk["section_title"],
                        "source_url": best_chunk["source_url"],
                        "pages": best_chunk["pages"],
                        "content_preview": best_chunk["content"][:200] + "...",
                        "similarity_score": avg_similarity,
                        "confidence_score": best_chunk["confidence_score"],
                        "chunk_count": len(chunks)
                    }
                    citations.append(citation)
                    citation_id += 1

            # Sort citations by similarity score
            citations.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Prepare summary
            total_chunks = len(search_results)
            avg_similarity = sum(r["similarity_score"] for r in search_results) / total_chunks if total_chunks > 0 else 0

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "success": True,
                "query": query,
                "ticker_filter": ticker,
                "section_filter": section_filter,
                "total_results": total_chunks,
                "citations": citations[:10],  # Limit to top 10 citations
                "summary": {
                    "documents_found": len(grouped_results),
                    "sections_found": sum(len(sections) for sections in grouped_results.values()),
                    "average_similarity": avg_similarity,
                    "processing_time": processing_time
                },
                "raw_results": search_results  # Include raw results for detailed analysis
            }

            logger.info(f"✅ Query completed: {total_chunks} results, {len(citations)} citations")
            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Query pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": processing_time
            }

    async def _create_collection_if_not_exists(self) -> bool:
        """Create the collection if it doesn't exist."""
        try:
            # Check if collection exists using a more robust method
            try:
                existing_collection = self.client.collections.get(self.collection_name)
                if existing_collection:
                    logger.info(f"📦 Collection '{self.collection_name}' already exists")
                    return True
            except Exception:
                # Collection doesn't exist, we'll create it
                pass

            # Create collection with proper schema
            logger.info(f"📦 Creating Weaviate collection: {self.collection_name}")

            # Create collection using Weaviate v4 API with proper configuration
            from weaviate.classes.config import Configure, Property, DataType

            try:
                self.client.collections.create(
                    name=self.collection_name,
                    description="HKEX Document sections with embeddings for semantic search",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Main content of the document chunk"),
                        Property(name="ticker", data_type=DataType.TEXT, description="Hong Kong stock ticker"),
                        Property(name="document_title", data_type=DataType.TEXT, description="Title of the source document"),
                        Property(name="section_title", data_type=DataType.TEXT, description="Section title"),
                        Property(name="source_url", data_type=DataType.TEXT, description="URL of the source document"),
                        Property(name="page_number", data_type=DataType.INT, description="Primary page number for this content chunk"),
                        Property(name="pages", data_type=DataType.INT_ARRAY, description="Page numbers where this content appears"),
                        Property(name="confidence_score", data_type=DataType.NUMBER, description="Confidence score for content extraction"),
                        Property(name="content_type", data_type=DataType.TEXT, description="Type of content: page_content or section_content"),
                        Property(name="extraction_method", data_type=DataType.TEXT, description="Method used for text extraction"),
                        Property(name="processed_date", data_type=DataType.TEXT, description="Date when document was processed"),
                        Property(name="chunk_id", data_type=DataType.INT, description="Chunk identifier within document"),
                        Property(name="char_count", data_type=DataType.INT, description="Character count of the chunk")
                    ]
                    # No vectorizer - we provide our own embeddings
                )
                logger.info(f"✅ Created Weaviate collection: {self.collection_name}")
            except Exception as create_error:
                # Check if error is because collection already exists
                if "already exists" in str(create_error):
                    logger.info(f"📦 Collection '{self.collection_name}' already exists (confirmed)")
                else:
                    logger.error(f"❌ Failed to create collection: {create_error}")
                    return False

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            return False
