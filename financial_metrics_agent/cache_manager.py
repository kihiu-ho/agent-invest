#!/usr/bin/env python3
"""
Enhanced PostgreSQL Cache Manager for Financial Data

Provides intelligent caching for financial data with TTL expiration,
graceful fallback handling, and optimized database operations.
"""

import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    
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
        load_dotenv()
        
except ImportError:
    pass

# Database imports with error handling
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Data class for cache entries."""
    ticker: str
    data_type: str
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    default_ttl_hours: int = 24
    max_retries: int = 3
    connection_timeout: int = 30
    pool_min_size: int = 2
    pool_max_size: int = 10
    enable_compression: bool = True

class FinancialDataCacheManager:
    """
    Enhanced PostgreSQL cache manager for financial data.
    
    Features:
    - Intelligent TTL-based caching
    - Graceful fallback on database failures
    - Optimized batch operations
    - Data compression for large payloads
    - Comprehensive error handling
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the cache manager.
        
        Args:
            config: Cache configuration options
        """
        self.config = config or CacheConfig()
        self.database_url = os.getenv("DATABASE_URL")
        self.connection_pool = None
        self.available = ASYNCPG_AVAILABLE and bool(self.database_url)
        
        # Task tracking for graceful shutdown
        self.pending_tasks = set()
        self.shutdown_flag = False
        self._shutdown_lock = asyncio.Lock()
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        
        if not self.available:
            if not ASYNCPG_AVAILABLE:
                logger.warning("⚠️ Cache disabled: asyncpg not available")
            elif not self.database_url:
                logger.warning("⚠️ Cache disabled: DATABASE_URL not configured")
        else:
            logger.info("✅ Financial data cache manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection pool and create tables."""
        if not self.available:
            return False
        
        try:
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.connection_timeout
            )
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("✅ Cache database connection pool initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache database: {e}")
            self.available = False
            return False
    
    async def _create_tables(self):
        """Create the comprehensive financial data cache tables."""
        if not self.connection_pool:
            return

        # Main financial data cache table (for general data)
        main_cache_sql = """
        CREATE TABLE IF NOT EXISTS financial_data_cache (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            compressed BOOLEAN DEFAULT FALSE,
            UNIQUE(ticker, data_type)
        );

        CREATE INDEX IF NOT EXISTS idx_financial_cache_ticker_type
        ON financial_data_cache(ticker, data_type);

        CREATE INDEX IF NOT EXISTS idx_financial_cache_expires
        ON financial_data_cache(expires_at);
        """

        # Historical price data table (individual daily records)
        price_data_sql = """
        CREATE TABLE IF NOT EXISTS yahoo_finance_price_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open_price DECIMAL(15,4),
            high_price DECIMAL(15,4),
            low_price DECIMAL(15,4),
            close_price DECIMAL(15,4),
            adjusted_close DECIMAL(15,4),
            volume BIGINT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            UNIQUE(ticker, date)
        );

        CREATE INDEX IF NOT EXISTS idx_price_data_ticker_date
        ON yahoo_finance_price_data(ticker, date);

        CREATE INDEX IF NOT EXISTS idx_price_data_expires
        ON yahoo_finance_price_data(expires_at);
        """

        # Historical data cache table (complete datasets)
        historical_cache_sql = """
        CREATE TABLE IF NOT EXISTS yahoo_finance_historical_cache (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            download_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            historical_data JSONB NOT NULL,
            data_points INTEGER NOT NULL,
            period VARCHAR(10) NOT NULL,
            start_date TIMESTAMP WITH TIME ZONE,
            end_date TIMESTAMP WITH TIME ZONE,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            UNIQUE(ticker, period)
        );

        CREATE INDEX IF NOT EXISTS idx_historical_cache_ticker_period
        ON yahoo_finance_historical_cache(ticker, period);

        CREATE INDEX IF NOT EXISTS idx_historical_cache_download_date
        ON yahoo_finance_historical_cache(download_date);

        CREATE INDEX IF NOT EXISTS idx_historical_cache_expires
        ON yahoo_finance_historical_cache(expires_at);
        """

        # Historical data cache table (complete datasets)
        historical_cache_sql = """
        CREATE TABLE IF NOT EXISTS yahoo_finance_historical_cache (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            download_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            historical_data JSONB NOT NULL,
            data_points INTEGER NOT NULL,
            period VARCHAR(10) NOT NULL,
            start_date TIMESTAMP WITH TIME ZONE,
            end_date TIMESTAMP WITH TIME ZONE,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            UNIQUE(ticker, period)
        );

        CREATE INDEX IF NOT EXISTS idx_yahoo_price_ticker_date
        ON yahoo_finance_price_data(ticker, date);

        CREATE INDEX IF NOT EXISTS idx_yahoo_price_ticker
        ON yahoo_finance_price_data(ticker);

        CREATE INDEX IF NOT EXISTS idx_yahoo_price_date
        ON yahoo_finance_price_data(date);

        CREATE INDEX IF NOT EXISTS idx_yahoo_price_expires
        ON yahoo_finance_price_data(expires_at);
        """

        # Financial metrics table
        metrics_data_sql = """
        CREATE TABLE IF NOT EXISTS yahoo_finance_metrics (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            date_retrieved TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            pe_ratio DECIMAL(10,2),
            market_cap BIGINT,
            dividend_yield DECIMAL(8,4),
            beta DECIMAL(8,4),
            revenue_growth DECIMAL(8,4),
            profit_margin DECIMAL(8,4),
            debt_to_equity DECIMAL(8,4),
            current_ratio DECIMAL(8,4),
            quick_ratio DECIMAL(8,4),
            return_on_equity DECIMAL(8,4),
            return_on_assets DECIMAL(8,4),
            price_to_book DECIMAL(8,4),
            price_to_sales DECIMAL(8,4),
            enterprise_value BIGINT,
            ebitda BIGINT,
            current_price DECIMAL(15,4),
            previous_close DECIMAL(15,4),
            day_high DECIMAL(15,4),
            day_low DECIMAL(15,4),
            fifty_two_week_high DECIMAL(15,4),
            fifty_two_week_low DECIMAL(15,4),
            volume BIGINT,
            avg_volume BIGINT,
            shares_outstanding BIGINT,
            float_shares BIGINT,
            earnings_per_share DECIMAL(8,4),
            book_value_per_share DECIMAL(8,4),
            cash_per_share DECIMAL(8,4),
            revenue_per_share DECIMAL(8,4),
            metadata JSONB,
            data_quality_score DECIMAL(5,2),
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            CONSTRAINT unique_ticker_date UNIQUE(ticker, date_retrieved)
        );

        CREATE INDEX IF NOT EXISTS idx_yahoo_metrics_ticker
        ON yahoo_finance_metrics(ticker);

        CREATE INDEX IF NOT EXISTS idx_yahoo_metrics_date
        ON yahoo_finance_metrics(date_retrieved);

        CREATE INDEX IF NOT EXISTS idx_yahoo_metrics_expires
        ON yahoo_finance_metrics(expires_at);
        """

        async with self.connection_pool.acquire() as conn:
            await conn.execute(main_cache_sql)
            await conn.execute(price_data_sql)
            await conn.execute(historical_cache_sql)
            await conn.execute(metrics_data_sql)
            logger.info("✅ Comprehensive financial data cache tables created/verified")
    
    async def get_cached_data(self, ticker: str, data_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data for a ticker and data type.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data (e.g., 'web_scraping', 'financial_metrics')
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        if not self.available or not self.connection_pool:
            return None
        
        try:
            # Handle Yahoo Finance data with specialized retrieval
            if data_type == "yahoo_finance":
                return await self._get_yahoo_finance_data(ticker)

            # Handle historical data with specialized retrieval
            if data_type == "historical_data":
                return await self._get_historical_data(ticker)

            # Default retrieval for other data types
            async with self.connection_pool.acquire() as conn:
                # Get cached data that hasn't expired
                result = await conn.fetchrow("""
                    SELECT data, metadata, created_at, expires_at, compressed
                    FROM financial_data_cache
                    WHERE ticker = $1 AND data_type = $2 AND expires_at > NOW()
                """, ticker, data_type)

                if result:
                    self.cache_hits += 1
                    # Handle data that's already a dictionary or convert from JSON
                    raw_data = result['data']
                    if isinstance(raw_data, dict):
                        data = raw_data.copy()
                    elif isinstance(raw_data, str):
                        import json
                        data = json.loads(raw_data)
                    else:
                        # Handle other formats by converting to dict if possible
                        try:
                            data = dict(raw_data)
                        except (TypeError, ValueError):
                            logger.error(f"❌ Cannot convert cached data to dictionary: {type(raw_data)}")
                            return None

                    # Add cache metadata
                    data['_cache_info'] = {
                        'cached_at': result['created_at'].isoformat(),
                        'expires_at': result['expires_at'].isoformat(),
                        'from_cache': True
                    }

                    logger.info(f"✅ Cache hit for {ticker}:{data_type}")
                    return data
                else:
                    self.cache_misses += 1
                    logger.info(f"ℹ️ Cache miss for {ticker}:{data_type}")
                    return None

        except Exception as e:
            self.cache_errors += 1
            logger.error(f"❌ Cache retrieval error for {ticker}:{data_type}: {e}")
            return None

    async def _get_yahoo_finance_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve Yahoo Finance data from specialized tables and combine.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Combined Yahoo Finance data or None if not found
        """
        try:
            combined_data = {
                'ticker': ticker,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }

            # Get financial metrics
            metrics = await self.get_latest_financial_metrics(ticker)
            if metrics:
                # Safely merge metrics data
                combined_data['financial_metrics'] = metrics.get('financial_metrics', {})
                combined_data['data_quality'] = metrics.get('data_quality', {})
                combined_data['date_retrieved'] = metrics.get('date_retrieved')
                combined_data['data_quality_score'] = metrics.get('data_quality_score')
                combined_data['has_financial_metrics'] = True
            else:
                combined_data['has_financial_metrics'] = False

            # Get recent historical data (last 30 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            historical_data = await self.get_historical_price_data(
                ticker, start_date.isoformat(), end_date.isoformat()
            )

            if historical_data:
                combined_data['historical_data'] = historical_data
                combined_data['has_historical_data'] = True

                # Add latest price info from historical data
                if historical_data:
                    latest = historical_data[-1]
                    combined_data['latest_price'] = {
                        'date': latest.get('date'),
                        'close': latest.get('close'),
                        'volume': latest.get('volume')
                    }
            else:
                combined_data['has_historical_data'] = False

            # Try to get from main cache as fallback
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT data, created_at, expires_at
                    FROM financial_data_cache
                    WHERE ticker = $1 AND data_type = 'yahoo_finance' AND expires_at > NOW()
                """, ticker)

                if result:
                    # Handle JSONB data consistently
                    main_cache_data = result['data']
                    if isinstance(main_cache_data, str):
                        main_cache_data = json.loads(main_cache_data)
                    elif not isinstance(main_cache_data, dict):
                        main_cache_data = dict(main_cache_data)

                    # Merge any additional data from main cache
                    for key, value in main_cache_data.items():
                        if key not in combined_data:
                            combined_data[key] = value

                    combined_data['_cache_info'] = {
                        'cached_at': result['created_at'].isoformat(),
                        'expires_at': result['expires_at'].isoformat(),
                        'from_cache': True,
                        'source': 'specialized_tables'
                    }

            # Return data if we have either metrics or historical data
            if combined_data.get('has_financial_metrics') or combined_data.get('has_historical_data'):
                self.cache_hits += 1
                logger.info(f"✅ Retrieved Yahoo Finance data for {ticker} from specialized tables")
                return combined_data
            else:
                self.cache_misses += 1
                logger.info(f"ℹ️ No Yahoo Finance data found for {ticker}")
                return None

        except Exception as e:
            logger.error(f"❌ Error retrieving Yahoo Finance data for {ticker}: {e}")
            self.cache_errors += 1
            return None

    async def _get_historical_data(self, ticker: str, period: str = "1Y") -> Optional[Dict[str, Any]]:
        """
        Retrieve cached historical data for a ticker and period.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., '1Y', '6M')

        Returns:
            Cached historical data if available and not expired, None otherwise
        """
        try:
            async with self.connection_pool.acquire() as conn:
                # Get cached historical data that hasn't expired
                result = await conn.fetchrow("""
                    SELECT historical_data, data_points, download_date,
                           start_date, end_date, metadata, created_at, expires_at
                    FROM yahoo_finance_historical_cache
                    WHERE ticker = $1 AND period = $2 AND expires_at > NOW()
                    ORDER BY download_date DESC
                    LIMIT 1
                """, ticker, period)

                if result:
                    self.cache_hits += 1

                    # Reconstruct the data structure
                    historical_data = result['historical_data']
                    if isinstance(historical_data, str):
                        historical_data = json.loads(historical_data)
                    elif not isinstance(historical_data, dict):
                        historical_data = dict(historical_data)

                    # Add cache metadata
                    cache_info = {
                        'cached_at': result['created_at'].isoformat(),
                        'expires_at': result['expires_at'].isoformat(),
                        'download_date': result['download_date'].isoformat(),
                        'data_points': result['data_points'],
                        'from_cache': True,
                        'cache_source': 'postgresql'
                    }

                    # Add cache info to the data
                    historical_data['_cache_info'] = cache_info

                    logger.info(f"✅ PostgreSQL cache hit for historical data {ticker}:{period} ({result['data_points']} points)")
                    return {"historical_data": historical_data}
                else:
                    self.cache_misses += 1
                    logger.info(f"ℹ️ PostgreSQL cache miss for historical data {ticker}:{period}")
                    return None

        except Exception as e:
            self.cache_errors += 1
            logger.error(f"❌ Historical data retrieval error for {ticker}:{period}: {e}")
            return None

    async def store_cached_data(self, ticker: str, data_type: str, data: Dict[str, Any],
                              ttl_hours: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data in cache with TTL expiration.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data
            data: Data to cache
            ttl_hours: Time to live in hours (default: 24)
            metadata: Optional metadata
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.available or not self.connection_pool:
            return False
        
        ttl_hours = ttl_hours or self.config.default_ttl_hours
        from datetime import timezone
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        
        try:
            # Remove cache metadata from data before storing
            clean_data = {k: v for k, v in data.items() if k != '_cache_info'}

            # Route Yahoo Finance data to specialized tables
            if data_type == "yahoo_finance":
                return await self._store_yahoo_finance_data(ticker, clean_data, ttl_hours, metadata)

            # Route historical data to specialized table
            if data_type == "historical_data":
                return await self._store_historical_data(ticker, clean_data, ttl_hours, metadata)

            # Default storage for other data types
            async with self.connection_pool.acquire() as conn:
                # Use UPSERT to handle duplicates
                await conn.execute("""
                    INSERT INTO financial_data_cache (ticker, data_type, data, metadata, expires_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (ticker, data_type)
                    DO UPDATE SET
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW(),
                        expires_at = EXCLUDED.expires_at
                """, ticker, data_type, json.dumps(clean_data),
                json.dumps(metadata) if metadata else None, expires_at)

                logger.info(f"✅ Cached data for {ticker}:{data_type} (expires: {expires_at})")
                return True

        except Exception as e:
            self.cache_errors += 1
            logger.error(f"❌ Cache storage error for {ticker}:{data_type}: {e}")
            return False

    async def _store_yahoo_finance_data(self, ticker: str, data: Dict[str, Any],
                                      ttl_hours: int, metadata: Optional[Dict[str, Any]]) -> bool:
        """
        Store Yahoo Finance data in appropriate specialized tables.

        Args:
            ticker: Stock ticker symbol
            data: Yahoo Finance data
            ttl_hours: Time to live in hours
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            success_count = 0
            total_operations = 0

            # Store historical price data if available
            historical_data = data.get('historical_data', [])
            if historical_data:
                total_operations += 1
                if await self.store_historical_price_data(ticker, historical_data, ttl_hours):
                    success_count += 1
                    logger.info(f"✅ Stored historical price data for {ticker}")
                else:
                    logger.warning(f"⚠️ Failed to store historical price data for {ticker}")

            # Store financial metrics if available
            if 'financial_metrics' in data or any(key in data for key in ['pe_ratio', 'market_cap', 'dividend_yield']):
                total_operations += 1
                if await self.store_financial_metrics(ticker, data, ttl_hours):
                    success_count += 1
                    logger.info(f"✅ Stored financial metrics for {ticker}")
                else:
                    logger.warning(f"⚠️ Failed to store financial metrics for {ticker}")

            # Also store in main cache as backup
            total_operations += 1
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO financial_data_cache (ticker, data_type, data, metadata, expires_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (ticker, data_type)
                    DO UPDATE SET
                        data = EXCLUDED.data,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW(),
                        expires_at = EXCLUDED.expires_at
                """, ticker, "yahoo_finance", json.dumps(data),
                json.dumps(metadata) if metadata else None, expires_at)
                success_count += 1

            # Return True if at least one operation succeeded
            if success_count > 0:
                logger.info(f"✅ Yahoo Finance data storage: {success_count}/{total_operations} operations successful for {ticker}")
                return True
            else:
                logger.error(f"❌ All Yahoo Finance data storage operations failed for {ticker}")
                return False

        except Exception as e:
            logger.error(f"❌ Error in Yahoo Finance data storage for {ticker}: {e}")
            return False
    
    async def invalidate_cache(self, ticker: str, data_type: Optional[str] = None) -> bool:
        """
        Invalidate cached data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Specific data type to invalidate (None for all)
            
        Returns:
            True if invalidated successfully
        """
        if not self.available or not self.connection_pool:
            return False
        
        try:
            async with self.connection_pool.acquire() as conn:
                if data_type:
                    await conn.execute("""
                        DELETE FROM financial_data_cache 
                        WHERE ticker = $1 AND data_type = $2
                    """, ticker, data_type)
                    logger.info(f"✅ Invalidated cache for {ticker}:{data_type}")
                else:
                    await conn.execute("""
                        DELETE FROM financial_data_cache WHERE ticker = $1
                    """, ticker)
                    logger.info(f"✅ Invalidated all cache for {ticker}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Cache invalidation error for {ticker}: {e}")
            return False

    async def store_historical_price_data(self, ticker: str, price_data: List[Dict[str, Any]],
                                        ttl_hours: int = 168) -> bool:  # 7 days default
        """
        Store historical price data in dedicated table with batch insertion.

        Args:
            ticker: Stock ticker symbol
            price_data: List of price data dictionaries with date, open, high, low, close, volume
            ttl_hours: Time to live in hours (default: 7 days)

        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.connection_pool or not price_data:
            return False

        try:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)

            # Prepare batch data with validation
            batch_data = []
            for item in price_data:
                # Validate price data
                if not self._validate_price_data(item):
                    logger.warning(f"⚠️ Invalid price data for {ticker}: {item}")
                    continue

                # Convert date string to date object
                date_str = item.get('date')
                if isinstance(date_str, str):
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    date_obj = date_str

                batch_data.append((
                    ticker,
                    date_obj,
                    float(item.get('open', 0)) if item.get('open') else None,
                    float(item.get('high', 0)) if item.get('high') else None,
                    float(item.get('low', 0)) if item.get('low') else None,
                    float(item.get('close', 0)) if item.get('close') else None,
                    float(item.get('adjusted_close', 0)) if item.get('adjusted_close') else None,
                    int(item.get('volume', 0)) if item.get('volume') else None,
                    expires_at
                ))

            if not batch_data:
                logger.warning(f"⚠️ No valid price data to store for {ticker}")
                return False

            async with self.connection_pool.acquire() as conn:
                # Use batch insert with conflict resolution
                await conn.executemany("""
                    INSERT INTO yahoo_finance_price_data
                    (ticker, date, open_price, high_price, low_price, close_price,
                     adjusted_close, volume, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (ticker, date)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        adjusted_close = EXCLUDED.adjusted_close,
                        volume = EXCLUDED.volume,
                        expires_at = EXCLUDED.expires_at,
                        created_at = NOW()
                """, batch_data)

                logger.info(f"✅ Stored {len(batch_data)} price records for {ticker}")
                return True

        except Exception as e:
            logger.error(f"❌ Error storing price data for {ticker}: {e}")
            self.cache_errors += 1
            return False

    async def store_financial_metrics(self, ticker: str, metrics: Dict[str, Any],
                                    ttl_hours: int = 24) -> bool:
        """
        Store financial metrics in dedicated table.

        Args:
            ticker: Stock ticker symbol
            metrics: Dictionary of financial metrics
            ttl_hours: Time to live in hours (default: 24 hours)

        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.connection_pool or not metrics:
            return False

        try:
            expires_at = datetime.now() + timedelta(hours=ttl_hours)

            # Extract and validate metrics
            financial_data = metrics.get('financial_metrics', {})
            data_quality = metrics.get('data_quality', {})

            async with self.connection_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO yahoo_finance_metrics
                    (ticker, pe_ratio, market_cap, dividend_yield, beta, revenue_growth,
                     profit_margin, debt_to_equity, current_ratio, quick_ratio,
                     return_on_equity, return_on_assets, price_to_book, price_to_sales,
                     enterprise_value, ebitda, current_price, previous_close, day_high, day_low,
                     fifty_two_week_high, fifty_two_week_low, volume, avg_volume,
                     shares_outstanding, float_shares, earnings_per_share, book_value_per_share,
                     cash_per_share, revenue_per_share, metadata, data_quality_score, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                            $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33)
                    ON CONFLICT ON CONSTRAINT unique_ticker_date
                    DO UPDATE SET
                        pe_ratio = EXCLUDED.pe_ratio,
                        market_cap = EXCLUDED.market_cap,
                        dividend_yield = EXCLUDED.dividend_yield,
                        beta = EXCLUDED.beta,
                        revenue_growth = EXCLUDED.revenue_growth,
                        profit_margin = EXCLUDED.profit_margin,
                        debt_to_equity = EXCLUDED.debt_to_equity,
                        current_ratio = EXCLUDED.current_ratio,
                        quick_ratio = EXCLUDED.quick_ratio,
                        return_on_equity = EXCLUDED.return_on_equity,
                        return_on_assets = EXCLUDED.return_on_assets,
                        price_to_book = EXCLUDED.price_to_book,
                        price_to_sales = EXCLUDED.price_to_sales,
                        enterprise_value = EXCLUDED.enterprise_value,
                        ebitda = EXCLUDED.ebitda,
                        current_price = EXCLUDED.current_price,
                        previous_close = EXCLUDED.previous_close,
                        day_high = EXCLUDED.day_high,
                        day_low = EXCLUDED.day_low,
                        fifty_two_week_high = EXCLUDED.fifty_two_week_high,
                        fifty_two_week_low = EXCLUDED.fifty_two_week_low,
                        volume = EXCLUDED.volume,
                        avg_volume = EXCLUDED.avg_volume,
                        shares_outstanding = EXCLUDED.shares_outstanding,
                        float_shares = EXCLUDED.float_shares,
                        earnings_per_share = EXCLUDED.earnings_per_share,
                        book_value_per_share = EXCLUDED.book_value_per_share,
                        cash_per_share = EXCLUDED.cash_per_share,
                        revenue_per_share = EXCLUDED.revenue_per_share,
                        metadata = EXCLUDED.metadata,
                        data_quality_score = EXCLUDED.data_quality_score,
                        expires_at = EXCLUDED.expires_at,
                        date_retrieved = NOW()
                """,
                ticker,
                self._safe_decimal(financial_data.get('pe_ratio')),
                self._safe_bigint(financial_data.get('market_cap')),
                self._safe_decimal(financial_data.get('dividend_yield')),
                self._safe_decimal(financial_data.get('beta')),
                self._safe_decimal(financial_data.get('revenue_growth')),
                self._safe_decimal(financial_data.get('profit_margin')),
                self._safe_decimal(financial_data.get('debt_to_equity')),
                self._safe_decimal(financial_data.get('current_ratio')),
                self._safe_decimal(financial_data.get('quick_ratio')),
                self._safe_decimal(financial_data.get('return_on_equity')),
                self._safe_decimal(financial_data.get('return_on_assets')),
                self._safe_decimal(financial_data.get('price_to_book')),
                self._safe_decimal(financial_data.get('price_to_sales')),
                self._safe_bigint(financial_data.get('enterprise_value')),
                self._safe_bigint(financial_data.get('ebitda')),
                self._safe_decimal(financial_data.get('current_price')),
                self._safe_decimal(financial_data.get('previous_close')),
                self._safe_decimal(financial_data.get('day_high')),
                self._safe_decimal(financial_data.get('day_low')),
                self._safe_decimal(financial_data.get('52_week_high')),
                self._safe_decimal(financial_data.get('52_week_low')),
                self._safe_bigint(financial_data.get('volume')),
                self._safe_bigint(financial_data.get('avg_volume')),
                self._safe_bigint(financial_data.get('shares_outstanding')),
                self._safe_bigint(financial_data.get('float_shares')),
                self._safe_decimal(financial_data.get('earnings_per_share')),
                self._safe_decimal(financial_data.get('book_value_per_share')),
                self._safe_decimal(financial_data.get('cash_per_share')),
                self._safe_decimal(financial_data.get('revenue_per_share')),
                json.dumps(data_quality) if data_quality else None,
                self._safe_decimal(data_quality.get('completeness_score')),
                expires_at
                )

                logger.info(f"✅ Stored financial metrics for {ticker}")
                return True

        except Exception as e:
            logger.error(f"❌ Error storing financial metrics for {ticker}: {e}")
            self.cache_errors += 1
            return False

    async def _store_historical_data(self, ticker: str, data: Dict[str, Any],
                                   ttl_hours: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store historical data in dedicated PostgreSQL table.

        Args:
            ticker: Stock ticker symbol
            data: Historical data dictionary
            ttl_hours: Time to live in hours
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.available or not self.connection_pool:
            return False

        try:
            # Extract historical data from the data structure
            historical_data = data.get('historical_data', {})
            if not historical_data:
                logger.warning(f"⚠️ No historical data found for {ticker}")
                return False

            # Calculate expiration time
            from datetime import timezone
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

            # Extract metadata from historical data
            data_points = historical_data.get('data_points', 0)
            period = historical_data.get('period', '1Y')
            start_date = None
            end_date = None

            # Parse start and end dates if available
            if historical_data.get('start_date'):
                try:
                    start_date = datetime.fromisoformat(historical_data['start_date'].replace('Z', '+00:00'))
                except Exception:
                    pass

            if historical_data.get('end_date'):
                try:
                    end_date = datetime.fromisoformat(historical_data['end_date'].replace('Z', '+00:00'))
                except Exception:
                    pass

            # Prepare metadata
            storage_metadata = {
                'api_source': 'yahoo_finance',
                'data_quality': data.get('data_quality', {}),
                'collection_time': data.get('collection_time', 0),
                'ticker': ticker,
                **(metadata or {})
            }

            async with self.connection_pool.acquire() as conn:
                # Use UPSERT to handle duplicates
                await conn.execute("""
                    INSERT INTO yahoo_finance_historical_cache
                    (ticker, historical_data, data_points, period, start_date, end_date,
                     metadata, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (ticker, period)
                    DO UPDATE SET
                        historical_data = EXCLUDED.historical_data,
                        data_points = EXCLUDED.data_points,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        metadata = EXCLUDED.metadata,
                        download_date = NOW(),
                        expires_at = EXCLUDED.expires_at,
                        created_at = NOW()
                """, ticker, json.dumps(historical_data), data_points, period,
                start_date, end_date, json.dumps(storage_metadata), expires_at)

                logger.info(f"✅ Stored historical data for {ticker}:{period} ({data_points} points) in PostgreSQL")
                return True

        except Exception as e:
            logger.error(f"❌ Error storing historical data for {ticker}: {e}")
            self.cache_errors += 1
            return False

    def _validate_price_data(self, data: Dict[str, Any]) -> bool:
        """Validate price data integrity."""
        try:
            # Check required fields
            if not data.get('date'):
                return False

            # Validate price values (no negative prices)
            price_fields = ['open', 'high', 'low', 'close', 'adjusted_close']
            for field in price_fields:
                value = data.get(field)
                if value is not None and (float(value) < 0 or float(value) > 1000000):
                    return False

            # Validate volume (non-negative)
            volume = data.get('volume')
            if volume is not None and int(volume) < 0:
                return False

            return True
        except (ValueError, TypeError):
            return False

    def _safe_decimal(self, value: Any) -> Optional[float]:
        """Safely convert value to decimal."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _safe_bigint(self, value: Any) -> Optional[int]:
        """Safely convert value to bigint."""
        if value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    async def get_historical_price_data(self, ticker: str, start_date: Optional[str] = None,
                                      end_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve historical price data for a ticker within date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format), optional
            end_date: End date (YYYY-MM-DD format), optional

        Returns:
            List of price data dictionaries or None if not found
        """
        if not self.available or not self.connection_pool:
            return None

        try:
            # Convert string dates to date objects
            start_date_obj = None
            end_date_obj = None

            if start_date:
                if isinstance(start_date, str):
                    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                else:
                    start_date_obj = start_date

            if end_date:
                if isinstance(end_date, str):
                    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
                else:
                    end_date_obj = end_date

            async with self.connection_pool.acquire() as conn:
                # Build query based on date parameters
                if start_date_obj and end_date_obj:
                    query = """
                        SELECT ticker, date, open_price, high_price, low_price, close_price,
                               adjusted_close, volume, created_at
                        FROM yahoo_finance_price_data
                        WHERE ticker = $1 AND date >= $2 AND date <= $3 AND expires_at > NOW()
                        ORDER BY date ASC
                    """
                    rows = await conn.fetch(query, ticker, start_date_obj, end_date_obj)
                elif start_date_obj:
                    query = """
                        SELECT ticker, date, open_price, high_price, low_price, close_price,
                               adjusted_close, volume, created_at
                        FROM yahoo_finance_price_data
                        WHERE ticker = $1 AND date >= $2 AND expires_at > NOW()
                        ORDER BY date ASC
                    """
                    rows = await conn.fetch(query, ticker, start_date_obj)
                else:
                    query = """
                        SELECT ticker, date, open_price, high_price, low_price, close_price,
                               adjusted_close, volume, created_at
                        FROM yahoo_finance_price_data
                        WHERE ticker = $1 AND expires_at > NOW()
                        ORDER BY date ASC
                    """
                    rows = await conn.fetch(query, ticker)

                if rows:
                    self.cache_hits += 1
                    price_data = []
                    for row in rows:
                        price_data.append({
                            'ticker': row['ticker'],
                            'date': row['date'].isoformat(),
                            'open': float(row['open_price']) if row['open_price'] else None,
                            'high': float(row['high_price']) if row['high_price'] else None,
                            'low': float(row['low_price']) if row['low_price'] else None,
                            'close': float(row['close_price']) if row['close_price'] else None,
                            'adjusted_close': float(row['adjusted_close']) if row['adjusted_close'] else None,
                            'volume': int(row['volume']) if row['volume'] else None,
                            'cached_at': row['created_at'].isoformat()
                        })

                    logger.info(f"✅ Retrieved {len(price_data)} price records for {ticker}")
                    return price_data
                else:
                    self.cache_misses += 1
                    return None

        except Exception as e:
            logger.error(f"❌ Error retrieving price data for {ticker}: {e}")
            self.cache_errors += 1
            return None

    async def get_latest_financial_metrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent financial metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of financial metrics or None if not found
        """
        if not self.available or not self.connection_pool:
            return None

        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM yahoo_finance_metrics
                    WHERE ticker = $1 AND expires_at > NOW()
                    ORDER BY date_retrieved DESC
                    LIMIT 1
                """, ticker)

                if row:
                    self.cache_hits += 1

                    # Convert row to dictionary with proper data types
                    metrics = {
                        'ticker': row['ticker'],
                        'date_retrieved': row['date_retrieved'].isoformat(),
                        'financial_metrics': {
                            'pe_ratio': float(row['pe_ratio']) if row['pe_ratio'] else None,
                            'market_cap': int(row['market_cap']) if row['market_cap'] else None,
                            'dividend_yield': float(row['dividend_yield']) if row['dividend_yield'] else None,
                            'beta': float(row['beta']) if row['beta'] else None,
                            'revenue_growth': float(row['revenue_growth']) if row['revenue_growth'] else None,
                            'profit_margin': float(row['profit_margin']) if row['profit_margin'] else None,
                            'debt_to_equity': float(row['debt_to_equity']) if row['debt_to_equity'] else None,
                            'current_ratio': float(row['current_ratio']) if row['current_ratio'] else None,
                            'quick_ratio': float(row['quick_ratio']) if row['quick_ratio'] else None,
                            'return_on_equity': float(row['return_on_equity']) if row['return_on_equity'] else None,
                            'return_on_assets': float(row['return_on_assets']) if row['return_on_assets'] else None,
                            'price_to_book': float(row['price_to_book']) if row['price_to_book'] else None,
                            'price_to_sales': float(row['price_to_sales']) if row['price_to_sales'] else None,
                            'enterprise_value': int(row['enterprise_value']) if row['enterprise_value'] else None,
                            'ebitda': int(row['ebitda']) if row['ebitda'] else None,
                            'current_price': float(row['current_price']) if row['current_price'] else None,
                            'previous_close': float(row['previous_close']) if row['previous_close'] else None,
                            'day_high': float(row['day_high']) if row['day_high'] else None,
                            'day_low': float(row['day_low']) if row['day_low'] else None,
                            '52_week_high': float(row['fifty_two_week_high']) if row['fifty_two_week_high'] else None,
                            '52_week_low': float(row['fifty_two_week_low']) if row['fifty_two_week_low'] else None,
                            'volume': int(row['volume']) if row['volume'] else None,
                            'avg_volume': int(row['avg_volume']) if row['avg_volume'] else None,
                            'shares_outstanding': int(row['shares_outstanding']) if row['shares_outstanding'] else None,
                            'float_shares': int(row['float_shares']) if row['float_shares'] else None,
                            'earnings_per_share': float(row['earnings_per_share']) if row['earnings_per_share'] else None,
                            'book_value_per_share': float(row['book_value_per_share']) if row['book_value_per_share'] else None,
                            'cash_per_share': float(row['cash_per_share']) if row['cash_per_share'] else None,
                            'revenue_per_share': float(row['revenue_per_share']) if row['revenue_per_share'] else None,
                        },
                        'data_quality': json.loads(row['metadata']) if row['metadata'] else {},
                        'data_quality_score': float(row['data_quality_score']) if row['data_quality_score'] else None,
                        '_cache_info': {
                            'cached_at': row['date_retrieved'].isoformat(),
                            'expires_at': row['expires_at'].isoformat(),
                            'from_cache': True
                        }
                    }

                    logger.info(f"✅ Retrieved financial metrics for {ticker}")
                    return metrics
                else:
                    self.cache_misses += 1
                    return None

        except Exception as e:
            logger.error(f"❌ Error retrieving financial metrics for {ticker}: {e}")
            self.cache_errors += 1
            return None

    async def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries from all tables.

        Returns:
            Number of entries cleaned up
        """
        if not self.available or not self.connection_pool:
            return 0

        try:
            total_count = 0
            async with self.connection_pool.acquire() as conn:
                # Clean up main cache table
                result1 = await conn.execute("""
                    DELETE FROM financial_data_cache WHERE expires_at <= NOW()
                """)
                count1 = int(result1.split()[-1]) if result1.split()[-1].isdigit() else 0

                # Clean up price data table
                result2 = await conn.execute("""
                    DELETE FROM yahoo_finance_price_data WHERE expires_at <= NOW()
                """)
                count2 = int(result2.split()[-1]) if result2.split()[-1].isdigit() else 0

                # Clean up metrics table
                result3 = await conn.execute("""
                    DELETE FROM yahoo_finance_metrics WHERE expires_at <= NOW()
                """)
                count3 = int(result3.split()[-1]) if result3.split()[-1].isdigit() else 0

                total_count = count1 + count2 + count3

                if total_count > 0:
                    logger.info(f"✅ Cleaned up {total_count} expired cache entries (cache: {count1}, price: {count2}, metrics: {count3})")

                return total_count

        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "available": self.available
        }
    
    async def close(self):
        """Close database connections and cleanup."""
        async with self._shutdown_lock:
            if self.shutdown_flag:
                return
            
            self.shutdown_flag = True
            
            # Wait for pending tasks
            if self.pending_tasks:
                logger.info(f"Waiting for {len(self.pending_tasks)} pending cache tasks...")
                await asyncio.gather(*self.pending_tasks, return_exceptions=True)
            
            # Close connection pool
            if self.connection_pool:
                await self.connection_pool.close()
                logger.info("✅ Cache database connections closed")
