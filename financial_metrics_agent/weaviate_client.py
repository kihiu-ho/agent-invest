"""
Weaviate Vector Database Client for HKEX Annual Report Integration

This module provides a comprehensive Weaviate client for querying and managing
HKEX annual report data in the vector database. It includes semantic search
capabilities, connection management, and error handling.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
import json

try:
    import weaviate
    from weaviate.classes.config import Configure
    from weaviate.classes.query import Filter, MetadataQuery
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    print("❌ Weaviate client not available. Install with: pip install weaviate-client")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def format_rfc3339_datetime(dt: datetime = None) -> str:
    """
    Format datetime to RFC3339 string compatible with Weaviate.

    Args:
        dt: Datetime object to format. If None, uses current UTC time.

    Returns:
        RFC3339 formatted string ending with 'Z' for UTC
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC and format as RFC3339
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

class WeaviateConnectionError(Exception):
    """Custom exception for Weaviate connection issues."""
    pass

class WeaviateQueryError(Exception):
    """Custom exception for Weaviate query issues."""
    pass

class WeaviateClient:
    """
    Enhanced Weaviate client for HKEX annual report data management.
    
    Features:
    - Automatic connection management with retry logic
    - Semantic search with configurable similarity thresholds
    - Document metadata tracking and freshness validation
    - Error handling and connection recovery
    - Support for batch operations and caching
    """
    
    def __init__(self):
        """Initialize Weaviate client with environment configuration."""
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate client not available. Install with: pip install weaviate-client")
        
        # Load configuration from environment
        self.api_key = os.getenv('WEAVIATE_API_KEY')
        self.url = os.getenv('WEAVIATE_URL')
        self.collection_name = os.getenv('WEAVIATE_COLLECTION_NAME', 'HKEXAnnualReports')
        self.similarity_threshold = float(os.getenv('WEAVIATE_SIMILARITY_THRESHOLD', '0.85'))
        self.cache_freshness_hours = int(os.getenv('WEAVIATE_CACHE_FRESHNESS_HOURS', '24'))
        self.vector_dimension = int(os.getenv('WEAVIATE_VECTOR_DIMENSION', '384'))
        self.embedding_model = os.getenv('WEAVIATE_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.max_retries = int(os.getenv('WEAVIATE_MAX_RETRIES', '3'))
        self.timeout_seconds = int(os.getenv('WEAVIATE_TIMEOUT_SECONDS', '60'))
        
        # Validate required configuration
        if not self.api_key or not self.url:
            raise ValueError("WEAVIATE_API_KEY and WEAVIATE_URL must be set in environment variables")
        
        self.client = None
        self.collection = None
        self._connection_attempts = 0
        self._retry_delay = 1.0

        # Flag to avoid repeated nearText errors when no vectorizer is configured
        self._near_text_unavailable = False

        logger.info(f"🔗 Weaviate client initialized for {self.url}")
        logger.info(f"📊 Collection: {self.collection_name}, Similarity threshold: {self.similarity_threshold}")
    
    async def connect(self) -> bool:
        """
        Establish connection to Weaviate with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                self._connection_attempts += 1
                
                # Create Weaviate client with improved connection settings
                import weaviate.classes.init as wvc

                # Enhanced Weaviate client configuration to handle gRPC issues
                try:
                    # Try with full configuration first
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=self.url,
                        auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                        skip_init_checks=True,  # Skip problematic gRPC health checks
                        additional_config=wvc.AdditionalConfig(
                            timeout=wvc.Timeout(
                                init=30,  # Reasonable init timeout
                                query=30,  # Reasonable query timeout
                                insert=60   # Longer insert timeout for batch operations
                            ),
                            startup_period=10  # Allow time for startup
                            # Note: ConnectionParams not available in this Weaviate version
                        )
                    )
                except Exception as config_error:
                    logger.warning(f"⚠️ Full configuration failed: {config_error}")
                    # Fallback to minimal configuration
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=self.url,
                        auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                        skip_init_checks=True  # Essential for avoiding gRPC health check issues
                    )
                
                # Test connection
                if self.client.is_ready():
                    # Get or create collection
                    try:
                        self.collection = self.client.collections.get(self.collection_name)
                        logger.info(f"✅ Connected to Weaviate collection '{self.collection_name}' (attempt {attempt + 1})")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Collection '{self.collection_name}' not found: {e}")
                        # Try to create collection
                        try:
                            await self._create_collection_if_not_exists()
                            self.collection = self.client.collections.get(self.collection_name)
                            logger.info(f"✅ Created and connected to Weaviate collection '{self.collection_name}' (attempt {attempt + 1})")
                            return True
                        except Exception as create_error:
                            logger.warning(f"⚠️ Could not create collection: {create_error}")
                            # Connection is valid even without collection
                            logger.info(f"✅ Connected to Weaviate successfully (attempt {attempt + 1})")
                            return True
                else:
                    raise WeaviateConnectionError("Weaviate client not ready")
                    
            except Exception as e:
                error_msg = str(e).lower()

                # Check for specific gRPC errors and handle them gracefully
                if "grpc" in error_msg or "health check" in error_msg:
                    logger.warning(f"⚠️ gRPC connection issue (attempt {attempt + 1}): {e}")
                else:
                    logger.warning(f"❌ Weaviate connection attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = min(self._retry_delay * (2 ** attempt), 10)  # Cap at 10 seconds
                    logger.info(f"⏳ Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    # On final attempt, try a simplified connection
                    logger.info("🔄 Final attempt with simplified configuration...")
                    try:
                        self.client = weaviate.connect_to_weaviate_cloud(
                            cluster_url=self.url,
                            auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                            skip_init_checks=True
                        )
                        if self.client.is_ready():
                            logger.info("✅ Connected with simplified configuration")
                            return True
                    except Exception as final_error:
                        logger.error(f"❌ Final simplified connection failed: {final_error}")

                    logger.error(f"❌ Failed to connect to Weaviate after {self.max_retries} attempts")
                    raise WeaviateConnectionError(f"Failed to connect to Weaviate: {e}")
        
        return False

    async def check_connection(self) -> Dict[str, Any]:
        """
        Check Weaviate database connectivity and return connection status with metadata.

        Returns:
            Dictionary with connection status, metadata, and any error information
        """
        try:
            # If no client exists, try to connect
            if not self.client:
                connection_success = await self.connect()
                if not connection_success:
                    return {
                        "connected": False,
                        "error": "Failed to establish initial connection",
                        "url": self.url,
                        "collection_name": self.collection_name,
                        "timestamp": datetime.now().isoformat()
                    }

            # Test if client is ready
            if not self.client.is_ready():
                return {
                    "connected": False,
                    "error": "Client not ready",
                    "url": self.url,
                    "collection_name": self.collection_name,
                    "timestamp": datetime.now().isoformat()
                }

            # Get cluster metadata
            cluster_metadata = {}
            try:
                meta = self.client.get_meta()
                cluster_metadata = {
                    "hostname": meta.get("hostname", "unknown"),
                    "version": meta.get("version", "unknown"),
                    "modules": meta.get("modules", {}),
                }
            except Exception as meta_error:
                logger.debug(f"Could not retrieve cluster metadata: {meta_error}")
                cluster_metadata = {"error": "Metadata unavailable"}

            # Check collection availability
            collection_status = {}
            try:
                if self.collection:
                    # Try a simple query to test collection functionality
                    test_response = self.collection.query.fetch_objects(limit=1)
                    collection_status = {
                        "available": True,
                        "name": self.collection_name,
                        "test_query_success": True
                    }
                else:
                    # Try to get collection
                    try:
                        test_collection = self.client.collections.get(self.collection_name)
                        collection_status = {
                            "available": True,
                            "name": self.collection_name,
                            "test_query_success": False,
                            "note": "Collection exists but not cached"
                        }
                    except Exception:
                        collection_status = {
                            "available": False,
                            "name": self.collection_name,
                            "error": "Collection not found"
                        }
            except Exception as collection_error:
                collection_status = {
                    "available": False,
                    "name": self.collection_name,
                    "error": str(collection_error)
                }

            return {
                "connected": True,
                "url": self.url,
                "collection_status": collection_status,
                "cluster_metadata": cluster_metadata,
                "configuration": {
                    "similarity_threshold": self.similarity_threshold,
                    "cache_freshness_hours": self.cache_freshness_hours,
                    "max_retries": self.max_retries,
                    "timeout_seconds": self.timeout_seconds
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Connection check failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "url": self.url,
                "collection_name": self.collection_name,
                "timestamp": datetime.now().isoformat()
            }

    def _ensure_proper_source_url(self, source_url: str, ticker: str) -> str:
        """
        Ensure source URL is in proper HTTP format.

        Args:
            source_url: Original source URL
            ticker: Stock ticker for fallback URL generation

        Returns:
            Properly formatted HTTP URL
        """
        if not source_url or source_url in ["local_file", "unknown", ""]:
            # Generate fallback HKEX URL
            clean_ticker = ticker.replace('.HK', '').zfill(4)
            return f"https://www.hkexnews.hk/listedco/{clean_ticker}/"

        # Ensure proper HTTP URL format
        if not source_url.startswith("http"):
            if source_url.startswith("/"):
                return f"https://www.hkexnews.hk{source_url}"
            else:
                clean_ticker = ticker.replace('.HK', '').zfill(4)
                return f"https://www.hkexnews.hk/listedco/{clean_ticker}/"

        return source_url

    async def disconnect(self):
        """Safely disconnect from Weaviate."""
        if self.client:
            try:
                self.client.close()
                logger.info("🔌 Disconnected from Weaviate")
            except Exception as e:
                logger.warning(f"⚠️ Error during Weaviate disconnection: {e}")
            finally:
                self.client = None
                self.collection = None
    
    async def search_documents(
        self,
        ticker: str,
        query: str,
        content_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Perform advanced semantic and hybrid search for HKEX annual report content.

        Args:
            ticker: Hong Kong stock ticker (e.g., '0005.HK')
            query: Search query for semantic matching
            content_types: Optional list of content types to filter by
            limit: Maximum number of results to return

        Returns:
            Dictionary with 'results' key containing list of matching documents with metadata and scores
        """
        if not self.client:
            await self.connect()
        
        if not self.collection:
            logger.warning(f"⚠️ Collection '{self.collection_name}' not available")
            return {"results": [], "success": False, "error": "Collection not available"}
        
        try:
            # Clean ticker format
            clean_ticker = ticker.replace('.HK', '').zfill(4)
            
            # Build filter conditions
            where_filter = Filter.by_property("ticker").equal(clean_ticker)
            
            # Add content type filter if specified
            if content_types:
                content_filter = Filter.by_property("content_type").contains_any(content_types)
                where_filter = where_filter & content_filter
            
            # Add freshness filter
            cutoff_date = datetime.now(timezone.utc) - timedelta(hours=self.cache_freshness_hours)
            freshness_filter = Filter.by_property("last_updated").greater_than(format_rfc3339_datetime(cutoff_date))
            where_filter = where_filter & freshness_filter
            
            # Try advanced search methods in order of preference
            response = None
            search_method = "none"

            if not self._near_text_unavailable:
                try:
                    # First try hybrid search (combines semantic + keyword search)
                    try:
                        response = self.collection.query.hybrid(
                            query=query,
                            filters=where_filter,
                            limit=limit,
                            return_metadata=MetadataQuery(score=True, creation_time=True),
                            return_properties=["content", "content_type", "section_title", "ticker",
                                              "document_title", "source_url", "last_updated", "confidence_score", "page_number"]
                        )
                        search_method = "hybrid"
                        logger.debug(f"🔍 Using hybrid search for query: '{query[:50]}...'")
                    except Exception as hybrid_error:
                        logger.debug(f"⚠️ Hybrid search failed: {hybrid_error}, trying semantic search")
                        # Fallback to semantic search
                        try:
                            response = self.collection.query.near_text(
                                query=query,
                                filters=where_filter,
                                limit=limit,
                                return_metadata=MetadataQuery(score=True, creation_time=True),
                                return_properties=["content", "content_type", "section_title", "ticker",
                                                  "document_title", "source_url", "last_updated", "confidence_score", "page_number"]
                            )
                            search_method = "semantic"
                            logger.debug(f"🔍 Using semantic search for query: '{query[:50]}...'")
                        except Exception as semantic_error:
                            logger.debug(f"⚠️ Semantic search also failed: {semantic_error}")
                            response = None
                            search_method = "fallback"
                except Exception as vectorizer_error:
                    # Detect vectorizer missing error and enable fallback
                    msg = str(vectorizer_error)
                    if ("VectorFromInput was called without vectorizer" in msg or
                        "could not vectorize input" in msg or
                        "nearText" in msg or "hybrid" in msg):
                        logger.warning(f"⚠️ Vectorizer unavailable (no vectorizer configured); falling back to filters-only retrieval")
                        self._near_text_unavailable = True
                        # Initialize response as None for fallback
                        response = None
                        search_method = "fallback"
                    else:
                        logger.error(f"❌ Unexpected search error: {vectorizer_error}")
                        raise vectorizer_error
            else:
                # Vectorizer already known to be unavailable
                response = None
                search_method = "fallback"

            # Process results if we have a valid response
            results = []
            if response and hasattr(response, 'objects') and response.objects:
                for obj in response.objects:
                    if obj.metadata.score >= self.similarity_threshold:
                        results.append({
                            "content": obj.properties.get("content", ""),
                            "content_type": obj.properties.get("content_type", ""),
                            "section_title": obj.properties.get("section_title", ""),
                            "ticker": obj.properties.get("ticker", ""),
                            "document_title": obj.properties.get("document_title", ""),
                            "source_url": obj.properties.get("source_url", ""),
                            "last_updated": obj.properties.get("last_updated", ""),
                            "confidence_score": obj.properties.get("confidence_score", 0.0),
                            "similarity_score": obj.metadata.score,
                            "creation_time": obj.metadata.creation_time
                        })

                logger.info(f"🔍 Found {len(results)} documents for {ticker} using {search_method} search")
                return {
                    "results": results,
                    "success": True,
                    "total_found": len(results),
                    "query": query,
                    "ticker": ticker,
                    "search_method": search_method
                }
            else:
                # No valid response, continue to fallback
                logger.debug(f"No valid response from vectorized search, trying fallback")

            # Fallback: filters-only retrieval by ticker and optional content types/freshness
            try:
                fallback_resp = self.collection.query.fetch_objects(
                    filters=where_filter,
                    limit=limit,
                    return_metadata=MetadataQuery(creation_time=True),
                    return_properties=["content", "content_type", "section_title", "ticker",
                                      "document_title", "source_url", "last_updated", "confidence_score", "page_number"]
                )
                results = []
                for obj in getattr(fallback_resp, 'objects', []) or []:
                    results.append({
                        "content": obj.properties.get("content", ""),
                        "content_type": obj.properties.get("content_type", ""),
                        "section_title": obj.properties.get("section_title", ""),
                        "ticker": obj.properties.get("ticker", ""),
                        "document_title": obj.properties.get("document_title", ""),
                        "source_url": obj.properties.get("source_url", ""),
                        "last_updated": obj.properties.get("last_updated", ""),
                        "confidence_score": obj.properties.get("confidence_score", 0.0),
                        # No semantic score available; set to None
                        "similarity_score": None,
                        "creation_time": obj.metadata.creation_time
                    })
                logger.info(f"🔍 Fallback retrieval returned {len(results)} documents for {ticker}")
                return {
                    "results": results,
                    "success": True,
                    "total_found": len(results),
                    "query": query,
                    "ticker": ticker,
                    "search_method": "fallback_filter"
                }
            except Exception as e:
                logger.error(f"❌ Fallback retrieval failed for {ticker}: {e}")
                return {"results": [], "success": False, "error": f"Fallback retrieval failed: {str(e)}"}

        except Exception as e:
            # Catch-all for any other errors in the main try block
            logger.error(f"❌ Unexpected error in search_documents for {ticker}: {e}")
            return {"results": [], "success": False, "error": f"Unexpected error: {str(e)}"}
    
    async def check_document_availability(self, ticker: str) -> Dict[str, Any]:
        """
        Check if documents are available for a given ticker.
        
        Args:
            ticker: Hong Kong stock ticker
            
        Returns:
            Dictionary with availability status and metadata
        """
        if not self.client:
            await self.connect()
        
        if not self.collection:
            return {
                "available": False,
                "ticker": ticker,
                "message": f"Collection '{self.collection_name}' not available"
            }
        
        try:
            clean_ticker = ticker.replace('.HK', '').zfill(4)
            
            # Query for any documents for this ticker
            try:
                response = self.collection.query.fetch_objects(
                    filters=Filter.by_property("ticker").equal(clean_ticker),
                    limit=1,
                    return_metadata=MetadataQuery(creation_time=True),
                    return_properties=["ticker", "document_title", "last_updated"]
                )
                # v4-style response has .objects
                if getattr(response, 'objects', None):
                    obj = response.objects[0]
                    return {
                        "available": True,
                        "ticker": obj.properties.get("ticker"),
                        "document_title": obj.properties.get("document_title"),
                        "last_updated": obj.properties.get("last_updated"),
                        "creation_time": obj.metadata.creation_time
                    }
                else:
                    return {
                        "available": False,
                        "ticker": clean_ticker,
                        "message": "No documents found in vector database"
                    }
            except TypeError:
                # Legacy client not supported; return unavailable to avoid raising
                return {
                    "available": False,
                    "ticker": clean_ticker,
                    "message": "No documents found in vector database (legacy query path not supported)"
                }
                
        except Exception as e:
            logger.error(f"❌ Error checking document availability for {ticker}: {e}")
            return {
                "available": False,
                "ticker": ticker,
                "error": str(e)
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def store_document_sections(
        self,
        ticker: str,
        sections: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store document sections in Weaviate with comprehensive metadata.

        Args:
            ticker: Hong Kong stock ticker
            sections: Dictionary of section data with metadata

        Returns:
            Dictionary containing storage results
        """
        if not self.client:
            await self.connect()



        try:
            # Ensure ticker is in correct "xxxx.HK" format
            if not ticker.endswith('.HK'):
                # If ticker is just numeric (e.g., "1299"), format it properly
                clean_ticker = ticker.replace('.HK', '').zfill(4)
                formatted_ticker = f"{clean_ticker}.HK"
            else:
                # Already in correct format, just ensure padding
                clean_ticker = ticker.replace('.HK', '').zfill(4)
                formatted_ticker = f"{clean_ticker}.HK"

            # Prepare objects for batch insertion
            objects = []
            for section_type, section_data in sections.items():
                # Ensure proper data types for Weaviate schema
                obj = {
                    "content": str(section_data.get("content", "")),
                    "content_type": str(section_data.get("content_type", section_type)),
                    "section_title": str(section_data.get("section_title", "")),
                    "ticker": str(formatted_ticker),  # Use standardized "xxxx.HK" format
                    "document_title": str(section_data.get("document_title", f"{formatted_ticker} Annual Report")),
                    "source_url": self._ensure_proper_source_url(
                        section_data.get("source_url", ""), formatted_ticker
                    ),
                    "last_updated": format_rfc3339_datetime(),
                    "confidence_score": float(section_data.get("confidence_score", 0.8)),
                    # Add page number if available
                    "page_number": int(section_data.get("page_number", 1)) if section_data.get("page_number") else 1
                }

                # Include embedding vector if available
                embedding = section_data.get("embedding")
                if embedding is not None and isinstance(embedding, (list, tuple)) and len(embedding) > 0:
                    # Ensure embedding is a list of floats with correct dimensions
                    try:
                        embedding_vector = [float(x) for x in embedding]
                        if len(embedding_vector) == self.vector_dimension:  # Should be 384 for all-MiniLM-L6-v2
                            obj["vector"] = embedding_vector
                            logger.debug(f"✅ Added {len(embedding_vector)}-dimensional embedding vector")
                        else:
                            logger.warning(f"⚠️ Embedding dimension mismatch: expected {self.vector_dimension}, got {len(embedding_vector)}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Invalid embedding format: {e}")
                else:
                    logger.debug(f"⚠️ No embedding available for section {section_type}")

                objects.append(obj)

            # Batch insert with fixed size and error checks per request
            stored_count = 0
            errors = []

            try:
                # Ensure we are using the correct collection handle
                collection = self.client.collections.use(self.collection_name)

                with collection.batch.fixed_size(batch_size=200) as batch:
                    for obj in objects:
                        # Separate vector from properties for Weaviate v4
                        vector = obj.pop("vector", None)

                        if vector is not None:
                            # Add object with explicit vector
                            batch.add_object(properties=obj, vector=vector)
                        else:
                            # Add object without vector (will use vectorizer if available)
                            batch.add_object(properties=obj)

                        # Stop early if too many errors
                        try:
                            if hasattr(batch, 'number_errors') and batch.number_errors and batch.number_errors > 10:
                                errors.append("Batch import stopped due to excessive errors (>10)")
                                break
                        except Exception:
                            pass

                # failed_objects may not be present in some client versions; guard it
                failed_objects = getattr(collection.batch, 'failed_objects', None)
                if failed_objects:
                    errors.append(f"Number of failed imports: {len(failed_objects)}. First: {failed_objects[0]}")
                else:
                    stored_count = len(objects)

                logger.info(f"✅ Stored {stored_count} document sections for {ticker}")

                return {
                    "success": True,
                    "sections_stored": stored_count,
                    "total_sections": len(objects),
                    "storage_method": "weaviate_vector_database",
                    "ticker": ticker,
                    "errors": errors if errors else None
                }

            except Exception as e:
                logger.error(f"❌ Batch storage error for {ticker}: {e}")
                return {
                    "success": False,
                    "error": f"Batch storage failed: {e}",
                    "ticker": ticker
                }

        except Exception as e:
            logger.error(f"❌ Error storing documents for {ticker}: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker
            }

    async def _create_collection_if_not_exists(self) -> bool:
        """Create the HKEX collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.collections.list_all()
            collection_names = [col.name for col in collections]

            if self.collection_name in collection_names:
                self.collection = self.client.collections.get(self.collection_name)
                return True

            # Create collection with proper schema and vectorizer
            logger.info(f"📦 Creating Weaviate collection: {self.collection_name}")

            # Import required classes for v4 API
            import weaviate.classes.config as wvc
            from weaviate.classes.config import Property, DataType

            # Create collection with vectorizer configuration
            try:
                # Try to create with text2vec-transformers (no API key required)
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    description="HKEX Annual Report sections with comprehensive metadata and semantic search",
                    vectorizer_config=wvc.Configure.Vectorizer.text2vec_transformers(),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Main content of the document section"),
                        Property(name="content_type", data_type=DataType.TEXT, description="Type of content (executive_summary, financial_highlights, etc.)"),
                        Property(name="section_title", data_type=DataType.TEXT, description="Title of the document section"),
                        Property(name="ticker", data_type=DataType.TEXT, description="Hong Kong stock ticker in xxxx.HK format"),
                        Property(name="document_title", data_type=DataType.TEXT, description="Title of the source document"),
                        Property(name="source_url", data_type=DataType.TEXT, description="URL of the source document"),
                        Property(name="last_updated", data_type=DataType.DATE, description="Last update timestamp"),
                        Property(name="confidence_score", data_type=DataType.NUMBER, description="Confidence score of the extraction"),
                        Property(name="page_number", data_type=DataType.INT, description="Page number in the source document")
                    ]
                )
                logger.info(f"✅ Created collection '{self.collection_name}' with text2vec-transformers vectorizer")

            except Exception as vectorizer_error:
                logger.warning(f"⚠️ Failed to create collection with vectorizer: {vectorizer_error}")
                logger.info("🔄 Trying to create collection without vectorizer...")

                # Fallback: Create collection without vectorizer
                self.collection = self.client.collections.create(
                    name=self.collection_name,
                    description="HKEX Annual Report sections (no vectorizer - keyword search only)",
                    properties=[
                        Property(name="content", data_type=DataType.TEXT, description="Main content of the document section"),
                        Property(name="content_type", data_type=DataType.TEXT, description="Type of content"),
                        Property(name="section_title", data_type=DataType.TEXT, description="Title of the document section"),
                        Property(name="ticker", data_type=DataType.TEXT, description="Hong Kong stock ticker in xxxx.HK format"),
                        Property(name="document_title", data_type=DataType.TEXT, description="Title of the source document"),
                        Property(name="source_url", data_type=DataType.TEXT, description="URL of the source document"),
                        Property(name="last_updated", data_type=DataType.DATE, description="Last update timestamp"),
                        Property(name="confidence_score", data_type=DataType.NUMBER, description="Confidence score"),
                        Property(name="page_number", data_type=DataType.INT, description="Page number in the source document")
                    ]
                )
                logger.info(f"✅ Created collection '{self.collection_name}' without vectorizer (keyword search only)")
                # Mark that vectorizer is unavailable
                self._near_text_unavailable = True

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            return False

        except Exception as e:
            logger.error(f"❌ Error creating collection: {e}")
            return False
