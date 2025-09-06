#!/usr/bin/env python3
"""
Database manager for Hong Kong web scraper caching system.
Handles PostgreSQL operations for storing and retrieving scraped content.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    from pathlib import Path

    # Try to load .env from current directory first, then parent directory
    env_paths = [
        Path('.env'),
        Path('../.env'),
        Path(__file__).parent.parent / '.env'
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()  # Fallback to default behavior

except ImportError:
    pass  # dotenv not available, rely on system environment

# Database imports with error handling
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    print("⚠️  asyncpg not available - database caching disabled")

# Import extraction constants
from extraction_constants import WebScrapingMethods

logger = logging.getLogger(__name__)

class WebScrapingDatabaseManager:
    """
    Database manager for web scraping content caching.
    Handles PostgreSQL operations for storing and retrieving scraped content.
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        self.connection_pool = None
        self.cache_duration_hours = 24  # Cache validity period
        self.available = ASYNCPG_AVAILABLE and bool(self.database_url)

        # Task tracking for graceful shutdown
        self.pending_storage_tasks = set()
        self.shutdown_flag = False
        self._shutdown_lock = asyncio.Lock()

        if not self.available:
            if not ASYNCPG_AVAILABLE:
                logger.warning("⚠️  Database caching disabled: asyncpg not available")
            elif not self.database_url:
                logger.warning("⚠️  Database caching disabled: DATABASE_URL not configured")
        else:
            logger.info("✅ Database manager initialized with caching enabled")
    
    async def initialize(self):
        """Initialize database connection pool and create tables if needed."""
        if not self.available:
            return False
        
        try:
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=5,
                command_timeout=30
            )
            
            # Create tables if they don't exist
            await self.create_tables()
            
            logger.info("✅ Database connection pool initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            self.available = False
            return False
    
    async def create_tables(self):
        """Create the web scraping cache table if it doesn't exist."""
        if not self.connection_pool:
            return
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS web_scraping_cache (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            url VARCHAR(500) NOT NULL,
            page_type VARCHAR(50) NOT NULL,
            source VARCHAR(50) NOT NULL,
            scraped_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            markdown_content TEXT,
            content_length INTEGER,
            extraction_method VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(ticker, url, page_type, source)
        );
        
        CREATE INDEX IF NOT EXISTS idx_web_scraping_cache_ticker ON web_scraping_cache(ticker);
        CREATE INDEX IF NOT EXISTS idx_web_scraping_cache_date ON web_scraping_cache(scraped_date);
        CREATE INDEX IF NOT EXISTS idx_web_scraping_cache_lookup ON web_scraping_cache(ticker, url, page_type, source);
        """
        
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(create_table_sql)
            logger.info("✅ Database tables created/verified")
            
        except Exception as e:
            logger.error(f"❌ Failed to create database tables: {e}")
            raise

    def create_storage_task(self, content_data: Dict[str, Any]) -> Optional[asyncio.Task]:
        """
        Create a tracked background task for storing content.

        Args:
            content_data: Dictionary containing scraped content and metadata

        Returns:
            asyncio.Task if created successfully, None if shutting down
        """
        if self.shutdown_flag or not self.available:
            return None

        # Create the storage task
        task = asyncio.create_task(self._store_content_with_tracking(content_data))

        # Add to pending tasks set
        self.pending_storage_tasks.add(task)

        # Remove from set when done (success or failure)
        task.add_done_callback(self.pending_storage_tasks.discard)

        return task

    async def _store_content_with_tracking(self, content_data: Dict[str, Any]) -> bool:
        """
        Internal method to store content with proper error handling and tracking.

        Args:
            content_data: Dictionary containing scraped content and metadata

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Check if we're shutting down
            if self.shutdown_flag:
                logger.debug("🛑 Skipping storage - shutdown in progress")
                return False

            return await self.store_content(content_data)

        except Exception as e:
            # Don't log errors during shutdown to avoid noise
            if not self.shutdown_flag:
                logger.warning(f"⚠️  Background storage task failed: {e}")
            return False

    async def get_cached_content(self, ticker: str, url: str, page_type: str, source: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached content if it exists and is recent.
        
        Args:
            ticker: Stock ticker symbol
            url: Target URL
            page_type: Type of page
            source: Website source
            
        Returns:
            Cached content dictionary or None if not found/expired
        """
        if not self.available or not self.connection_pool:
            return None
        
        try:
            # Calculate cutoff time for cache validity
            cutoff_time = datetime.now() - timedelta(hours=self.cache_duration_hours)
            
            query = """
            SELECT ticker, url, page_type, source, scraped_date, markdown_content, 
                   content_length, extraction_method
            FROM web_scraping_cache 
            WHERE ticker = $1 AND url = $2 AND page_type = $3 AND source = $4 
                  AND scraped_date > $5
            ORDER BY scraped_date DESC 
            LIMIT 1
            """
            
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(query, ticker, url, page_type, source, cutoff_time)
                
                if row:
                    logger.info(f"✅ Cache hit for {source} {page_type} - {ticker}")
                    return {
                        "success": True,
                        "url": row['url'],
                        "page_type": row['page_type'],
                        "source": row['source'],
                        "ticker": row['ticker'],
                        "markdown_content": row['markdown_content'],
                        "content_length": row['content_length'],
                        "scraped_at": row['scraped_date'].timestamp(),
                        "extraction_method": row['extraction_method'],
                        "cached": True,
                        "has_meaningful_content": bool(row['markdown_content'] and len(row['markdown_content']) > 100)
                    }
                else:
                    logger.debug(f"🔍 Cache miss for {source} {page_type} - {ticker}")
                    return None
                    
        except Exception as e:
            logger.warning(f"⚠️  Database cache lookup failed for {source} {page_type} - {ticker}: {e}")
            return None
    
    async def store_content(self, content_data: Dict[str, Any]) -> bool:
        """
        Store scraped content in the database cache.
        
        Args:
            content_data: Dictionary containing scraped content and metadata
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.available or not self.connection_pool:
            return False
        
        try:
            # Extract required fields
            ticker = content_data.get('ticker')
            url = content_data.get('url')
            page_type = content_data.get('page_type')
            source = content_data.get('source')
            markdown_content = content_data.get('markdown_content', '')
            content_length = content_data.get('content_length', 0)
            extraction_method = content_data.get('extraction_method', WebScrapingMethods.UNKNOWN)
            
            if not all([ticker, url, page_type, source]):
                logger.warning("⚠️  Missing required fields for database storage")
                return False
            
            # Use UPSERT to handle duplicates
            upsert_query = """
            INSERT INTO web_scraping_cache 
                (ticker, url, page_type, source, markdown_content, content_length, extraction_method, scraped_date, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            ON CONFLICT (ticker, url, page_type, source) 
            DO UPDATE SET 
                markdown_content = EXCLUDED.markdown_content,
                content_length = EXCLUDED.content_length,
                extraction_method = EXCLUDED.extraction_method,
                scraped_date = EXCLUDED.scraped_date,
                updated_at = NOW()
            """
            
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    upsert_query,
                    ticker, url, page_type, source, markdown_content, content_length, extraction_method
                )
            
            logger.info(f"✅ Content cached for {source} {page_type} - {ticker} ({content_length:,} chars)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to store content in database cache: {e}")
            return False
    
    async def get_cache_stats(self, ticker: str = None) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Args:
            ticker: Optional ticker to filter stats
            
        Returns:
            Dictionary containing cache statistics
        """
        if not self.available or not self.connection_pool:
            return {"available": False}
        
        try:
            base_query = """
            SELECT 
                COUNT(*) as total_entries,
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(DISTINCT source) as unique_sources,
                AVG(content_length) as avg_content_length,
                MAX(scraped_date) as latest_scrape,
                MIN(scraped_date) as earliest_scrape
            FROM web_scraping_cache
            """
            
            params = []
            if ticker:
                base_query += " WHERE ticker = $1"
                params.append(ticker)
            
            async with self.connection_pool.acquire() as connection:
                row = await connection.fetchrow(base_query, *params)
                
                return {
                    "available": True,
                    "total_entries": row['total_entries'],
                    "unique_tickers": row['unique_tickers'],
                    "unique_sources": row['unique_sources'],
                    "avg_content_length": int(row['avg_content_length']) if row['avg_content_length'] else 0,
                    "latest_scrape": row['latest_scrape'],
                    "earliest_scrape": row['earliest_scrape'],
                    "cache_duration_hours": self.cache_duration_hours
                }
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to get cache statistics: {e}")
            return {"available": False, "error": str(e)}
    
    async def cleanup_old_cache(self, days_to_keep: int = 7) -> int:
        """
        Clean up old cache entries to prevent database bloat.
        
        Args:
            days_to_keep: Number of days of cache to retain
            
        Returns:
            Number of entries deleted
        """
        if not self.available or not self.connection_pool:
            return 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            delete_query = """
            DELETE FROM web_scraping_cache 
            WHERE scraped_date < $1
            """
            
            async with self.connection_pool.acquire() as connection:
                result = await connection.execute(delete_query, cutoff_date)
                
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1]) if result else 0
                
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} old cache entries (older than {days_to_keep} days)")
                
                return deleted_count
                
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup old cache entries: {e}")
            return 0
    
    async def close(self, timeout: float = 30.0):
        """
        Close database connection pool with graceful shutdown.

        Args:
            timeout: Maximum time to wait for pending operations (seconds)
        """
        async with self._shutdown_lock:
            if self.shutdown_flag:
                return  # Already shutting down

            logger.info("🛑 Starting graceful database shutdown...")

            # Set shutdown flag to prevent new storage tasks
            self.shutdown_flag = True

            # Wait for pending storage tasks to complete
            if self.pending_storage_tasks:
                pending_count = len(self.pending_storage_tasks)
                logger.info(f"⏳ Waiting for {pending_count} pending storage tasks...")

                try:
                    # Wait for all pending tasks with timeout
                    await asyncio.wait_for(
                        asyncio.gather(*self.pending_storage_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                    logger.info(f"✅ All pending storage tasks completed")

                except asyncio.TimeoutError:
                    logger.warning(f"⚠️  Timeout waiting for storage tasks - forcing shutdown")

                    # Cancel remaining tasks
                    for task in self.pending_storage_tasks:
                        if not task.done():
                            task.cancel()

                    # Wait a bit for cancellations to complete
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self.pending_storage_tasks, return_exceptions=True),
                            timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("⚠️  Some tasks did not cancel cleanly")

                except Exception as e:
                    logger.warning(f"⚠️  Error during task cleanup: {e}")

            # Close the connection pool
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None
                logger.info("✅ Database connection pool closed")

            # Clear task tracking
            self.pending_storage_tasks.clear()
            logger.info("✅ Database shutdown completed")

# Global database manager instance
db_manager = WebScrapingDatabaseManager()
