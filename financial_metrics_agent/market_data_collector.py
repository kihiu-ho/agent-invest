"""
Market Data Collector

Handles Yahoo Finance API integration for retrieving financial metrics,
historical price data, and company information with comprehensive error handling.
"""

import asyncio
import logging
import re
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import random
import json
from requests.exceptions import HTTPError, RequestException, Timeout
import requests
from citation_tracker import CitationTracker
from historical_data_cache import HistoricalDataCache

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    Collects financial data from Yahoo Finance API with async support and error handling.
    """
    
    def __init__(self, max_workers: int = 5, request_delay: float = 2.0, max_retries: int = 3,
                 enable_cache: bool = True, cache_expiry_hours: int = 24, cache_manager=None):
        """
        Initialize the market data collector with enhanced rate limiting and error handling.

        Args:
            max_workers: Maximum number of concurrent requests
            request_delay: Delay between requests to avoid rate limiting (increased default)
            max_retries: Maximum number of retry attempts for failed requests
            enable_cache: Enable historical data caching
            cache_expiry_hours: Hours after which cached data expires
            cache_manager: Optional cache manager for PostgreSQL storage
        """
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.citation_tracker = CitationTracker()

        # Initialize caching systems
        self.enable_cache = enable_cache
        self.cache_manager = cache_manager  # PostgreSQL cache (primary)
        self.historical_cache = HistoricalDataCache(cache_expiry_hours=cache_expiry_hours) if enable_cache else None  # File cache (secondary)

        # Rate limiting and circuit breaker state
        self.last_request_time = 0
        self.consecutive_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = 300  # 5 minutes
        self.circuit_breaker_open_time = None

        # Request session for connection pooling and better error handling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Supported time periods for historical data
        self.time_periods = {
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y",
            "6M": "6mo",
            "1M": "1mo",
            "YTD": "ytd"
        }

        cache_status = "✅" if enable_cache else "❌"
        cache_status = "✅" if enable_cache else "❌"
        pg_cache_status = "✅" if cache_manager and cache_manager.available else "❌"
        logger.info(f"MarketDataCollector initialized with {max_workers} workers, {request_delay}s delay, {max_retries} retries")
        logger.info(f"📊 Historical data caching: {cache_status} (expiry: {cache_expiry_hours}h)")
        logger.info(f"🗄️ PostgreSQL cache: {pg_cache_status}, File cache: {cache_status}")
        logger.info(f"📊 Historical data caching: {cache_status} (expiry: {cache_expiry_hours}h)")

    async def _rate_limit_delay(self):
        """Implement rate limiting with jitter to avoid thundering herd."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            delay = self.request_delay - time_since_last
            # Add jitter to prevent synchronized requests
            jitter = random.uniform(0.1, 0.5)
            total_delay = delay + jitter
            logger.debug(f"Rate limiting: waiting {total_delay:.2f}s")
            await asyncio.sleep(total_delay)

        self.last_request_time = time.time()

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open due to consecutive failures."""
        if self.consecutive_failures < self.circuit_breaker_threshold:
            return False

        if self.circuit_breaker_open_time is None:
            self.circuit_breaker_open_time = time.time()
            logger.warning(f"🔴 Circuit breaker opened after {self.consecutive_failures} consecutive failures")
            return True

        # Check if reset time has passed
        if time.time() - self.circuit_breaker_open_time > self.circuit_breaker_reset_time:
            logger.info("🟡 Circuit breaker reset time reached, attempting to close")
            self.circuit_breaker_open_time = None
            self.consecutive_failures = 0
            return False

        return True

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        if self._is_circuit_breaker_open():
            raise Exception("Circuit breaker is open - too many consecutive failures")

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                await self._rate_limit_delay()
                result = await func(*args, **kwargs)

                # Success - reset failure counter
                if self.consecutive_failures > 0:
                    logger.info(f"✅ Request succeeded after {self.consecutive_failures} failures - resetting counter")
                    self.consecutive_failures = 0

                return result

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check for rate limiting or network errors
                if any(indicator in error_msg for indicator in ['429', 'too many requests', 'rate limit']):
                    self.consecutive_failures += 1
                    if attempt < self.max_retries:
                        # Exponential backoff with jitter for rate limiting
                        delay = min(60, (2 ** attempt) * 2 + random.uniform(1, 3))
                        logger.warning(f"⚠️ Rate limited (attempt {attempt + 1}/{self.max_retries + 1}), waiting {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)
                        continue
                elif any(indicator in error_msg for indicator in ['json', 'expecting value', 'connection', 'timeout']):
                    self.consecutive_failures += 1
                    if attempt < self.max_retries:
                        # Shorter backoff for JSON/connection errors
                        delay = min(10, (2 ** attempt) + random.uniform(0.5, 1.5))
                        logger.warning(f"⚠️ Network/JSON error (attempt {attempt + 1}/{self.max_retries + 1}), waiting {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)
                        continue
                else:
                    # Non-retryable error
                    logger.error(f"❌ Non-retryable error: {e}")
                    break

        # All retries exhausted
        self.consecutive_failures += 1
        logger.error(f"❌ All {self.max_retries + 1} attempts failed. Consecutive failures: {self.consecutive_failures}")
        raise last_exception
    
    async def collect_ticker_data(self, ticker: str, time_period: str = "1Y") -> Dict[str, Any]:
        """
        Collect comprehensive data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', '0700.HK')
            time_period: Time period for historical data (1Y, 2Y, 5Y, etc.)
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"🔍 Collecting data for ticker: {ticker}")
        start_time = time.time()

        # Start citation tracking for this ticker
        self.citation_tracker.start_analysis(ticker)

        try:
            # Validate time period
            if time_period not in self.time_periods:
                logger.warning(f"Invalid time period {time_period}, using 1Y")
                time_period = "1Y"

            # Three-tier caching strategy: PostgreSQL -> File Cache -> API
            cached_data = await self._get_cached_historical_data(ticker, time_period)

            if cached_data:
                # Determine cache source from data structure
                if 'historical_data' in cached_data:
                    # PostgreSQL cache format
                    cache_source = cached_data.get('historical_data', {}).get('_cache_info', {}).get('cache_source', 'postgresql')
                elif 'data' in cached_data:
                    # File cache format
                    cache_source = 'file_cache'
                else:
                    cache_source = 'unknown'

                logger.info(f"✅ Using cached historical data for {ticker} from {cache_source}")

                # Add fresh basic info and metrics to cached historical data
                stock = yf.Ticker(ticker)
                tasks = [
                    self._get_basic_info(stock, ticker),
                    self._get_financial_metrics(stock, ticker),
                    self._get_company_info(stock, ticker)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Combine cached historical data with fresh real-time data
                ticker_data = {
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "collection_time": time.time() - start_time,
                    "success": True,
                    "error": None,
                    "data_source": f"hybrid_{cache_source}_api"
                }

                # Add fresh data
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Task {i} failed for {ticker}: {result}")
                        continue
                    ticker_data.update(result)

                # Use cached historical data - normalize structure first
                if 'historical_data' in cached_data:
                    # PostgreSQL cache format: {"historical_data": {...}}
                    ticker_data.update(cached_data)
                elif 'data' in cached_data and 'historical_data' in cached_data['data']:
                    # File cache format: {"data": {"historical_data": {...}}}
                    ticker_data.update(cached_data['data'])
                else:
                    # Fallback: use cached_data as-is
                    ticker_data.update(cached_data)

                return ticker_data

            # No cache hit - collect all data from API
            logger.info(f"📡 Fetching fresh data from API for {ticker}")
            stock = yf.Ticker(ticker)

            # Collect data concurrently
            tasks = [
                self._get_basic_info(stock, ticker),
                self._get_financial_metrics(stock, ticker),
                self._get_historical_data_with_cache(stock, ticker, time_period),
                self._get_company_info(stock, ticker)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            ticker_data = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "collection_time": time.time() - start_time,
                "success": True,
                "error": None,
                "data_source": "fresh_api"
            }
            
            # Process results with enhanced data structure preservation
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in data collection task {i} for {ticker}: {result}")
                    ticker_data["success"] = False
                    ticker_data["error"] = str(result)
                else:
                    # Preserve enhanced financial metrics structure
                    if 'financial_metrics' in result and 'data_quality' in result:
                        # Enhanced financial metrics with data quality
                        ticker_data['financial_metrics'] = result['financial_metrics']
                        ticker_data['data_quality'] = result['data_quality']
                    else:
                        ticker_data.update(result)
            
            logger.info(f"✅ Data collection completed for {ticker} in {ticker_data['collection_time']:.2f}s")
            return ticker_data
            
        except Exception as e:
            logger.error(f"❌ Failed to collect data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "collection_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    async def collect_multiple_tickers(self, tickers: List[str], time_period: str = "1Y") -> Dict[str, Any]:
        """
        Collect data for multiple tickers concurrently.
        
        Args:
            tickers: List of ticker symbols
            time_period: Time period for historical data
            
        Returns:
            Dictionary with results for all tickers
        """
        logger.info(f"📊 Collecting data for {len(tickers)} tickers: {tickers}")
        start_time = time.time()
        
        # Process tickers with enhanced rate limiting and circuit breaker protection
        tasks = []
        for i, ticker in enumerate(tickers):
            # Check circuit breaker before each request
            if self._is_circuit_breaker_open():
                logger.error(f"🔴 Circuit breaker open - skipping remaining tickers starting with {ticker}")
                break

            if i > 0:
                # Progressive delay for multiple tickers to be more respectful
                delay = self.request_delay * (1 + i * 0.1)  # Increase delay slightly for each ticker
                await asyncio.sleep(delay)
            tasks.append(self.collect_ticker_data(ticker, time_period))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        ticker_results = {}
        successful_count = 0
        
        for i, result in enumerate(results):
            ticker = tickers[i]
            if isinstance(result, Exception):
                logger.error(f"Exception collecting data for {ticker}: {result}")
                ticker_results[ticker] = {
                    "ticker": ticker,
                    "success": False,
                    "error": str(result)
                }
            else:
                ticker_results[ticker] = result
                if result.get("success", False):
                    successful_count += 1
        
        total_time = time.time() - start_time
        logger.info(f"✅ Completed data collection: {successful_count}/{len(tickers)} successful in {total_time:.2f}s")
        
        return {
            "tickers": ticker_results,
            "summary": {
                "total_tickers": len(tickers),
                "successful": successful_count,
                "failed": len(tickers) - successful_count,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _get_basic_info(self, stock: yf.Ticker, ticker: str) -> Dict[str, Any]:
        """Get basic stock information with retry logic and enhanced error handling."""
        async def _fetch_info():
            loop = asyncio.get_event_loop()

            # Enhanced error handling for yfinance info retrieval
            try:
                info = await loop.run_in_executor(self.executor, self._safe_get_info, stock)

                # Validate JSON response
                if info is None:
                    logger.warning(f"No basic info available for {ticker} - trying alternative methods")
                    info = await self._get_alternative_basic_info(ticker)

                if not info:
                    info = {}

                return {
                    "basic_info": {
                        "symbol": info.get("symbol", ticker),
                        "long_name": info.get("longName", ""),
                        "short_name": info.get("shortName", ""),
                        "sector": info.get("sector", ""),
                        "industry": info.get("industry", ""),
                        "country": info.get("country", ""),
                        "currency": info.get("currency", ""),
                        "exchange": info.get("exchange", ""),
                        "website": info.get("website", "")
                    }
                }
            except Exception as e:
                # Re-raise for retry mechanism to handle
                raise Exception(f"Failed to get basic info for {ticker}: {e}")

        try:
            return await self._retry_with_backoff(_fetch_info)
        except Exception as e:
            logger.warning(f"All attempts failed for basic info {ticker}: {e}")
            return {"basic_info": {}}

    def _safe_get_info(self, stock: yf.Ticker) -> Dict[str, Any]:
        """Safely get stock info with enhanced error handling for JSON parsing."""
        try:
            info = stock.info

            # Validate that we got a proper dictionary response
            if not isinstance(info, dict):
                logger.warning(f"Invalid info response type: {type(info)}")
                return None

            # Check for empty or error responses
            # Use OR logic - if we have either regularMarketPrice OR symbol, the data is valid
            if not info or (info.get('regularMarketPrice') is None and info.get('symbol') is None and info.get('currentPrice') is None):
                logger.warning("Empty or invalid info response - no price or symbol data")
                return None

            return info

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in stock.info: {e}")
            raise Exception(f"JSON parsing failed: {e}")
        except HTTPError as e:
            if e.response.status_code == 429:
                raise Exception(f"Rate limited (429): {e}")
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except (RequestException, Timeout) as e:
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in _safe_get_info: {e}")
            raise Exception(f"Unexpected error: {e}")

    async def _get_alternative_basic_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic info using alternative methods for Hong Kong tickers."""
        try:
            # Try different ticker formats for HK stocks
            if ticker.endswith('.HK'):
                alternative_formats = [
                    ticker.replace('.HK', '.HKG'),  # Alternative format
                    f"{ticker.split('.')[0].zfill(4)}.HK",  # Ensure 4-digit format
                    f"HK:{ticker.split('.')[0]}",  # Bloomberg-style format
                ]

                for alt_ticker in alternative_formats:
                    try:
                        alt_stock = yf.Ticker(alt_ticker)
                        loop = asyncio.get_event_loop()
                        alt_info = await loop.run_in_executor(
                            self.executor,
                            self._safe_get_info,
                            alt_stock
                        )

                        if alt_info and alt_info.get('symbol'):
                            logger.info(f"✅ Retrieved basic info for {ticker} using format {alt_ticker}")
                            return alt_info

                    except Exception as e:
                        logger.debug(f"Failed alternative format {alt_ticker}: {e}")
                        continue

            # Try to get minimal info from historical data
            try:
                stock = yf.Ticker(ticker)
                loop = asyncio.get_event_loop()
                hist = await loop.run_in_executor(
                    self.executor,
                    lambda: stock.history(period="5d", timeout=10)
                )

                if not hist.empty:
                    latest = hist.iloc[-1]
                    logger.info(f"✅ Retrieved basic price info from history for {ticker}")
                    return {
                        'symbol': ticker,
                        'regularMarketPrice': latest.get('Close'),
                        'currency': 'HKD' if ticker.endswith('.HK') else 'USD'
                    }

            except Exception as e:
                logger.debug(f"Failed to get historical data for basic info {ticker}: {e}")

            return {}

        except Exception as e:
            logger.warning(f"All alternative methods failed for {ticker}: {e}")
            return {}
    
    async def _get_financial_metrics(self, stock: yf.Ticker, ticker: str) -> Dict[str, Any]:
        """Get comprehensive financial metrics and ratios with enhanced HK ticker support and retry logic."""
        async def _fetch_financial_metrics():
            try:
                loop = asyncio.get_event_loop()
                info = await loop.run_in_executor(self.executor, self._safe_get_info, stock)

                # Handle None info for Hong Kong tickers and other cases
                if info is None:
                    logger.warning(f"No financial info available for {ticker} - attempting enhanced collection")
                    info = await self._get_enhanced_hk_data(stock, ticker)

                # Track enhanced collection as a data source
                if info:
                    enhanced_metrics_count = len([v for v in info.values() if v is not None])
                    if enhanced_metrics_count > 0:
                        self.citation_tracker.track_yahoo_finance_data(
                            ticker, enhanced_metrics_count, f"yfinance.Ticker('{ticker}').info (enhanced collection)"
                        )

                # Extract comprehensive financial metrics with citation tracking
                financial_metrics = {}

                # Price metrics with citations
                price_metrics = {
                    "current_price": info.get("currentPrice"),
                    "previous_close": info.get("previousClose"),
                    "open_price": info.get("open"),
                    "day_high": info.get("dayHigh"),
                    "day_low": info.get("dayLow"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                }

                # Add price metrics to financial_metrics
                for metric_name, value in price_metrics.items():
                    financial_metrics[metric_name] = value

                # Market metrics with citations
                market_metrics = {
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                    "shares_short": info.get("sharesShort"),
                }

                # Add market metrics to financial_metrics
                for metric_name, value in market_metrics.items():
                    financial_metrics[metric_name] = value

                # Valuation ratios with citations
                valuation_metrics = {
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "peg_ratio": info.get("pegRatio"),
                "ev_revenue": info.get("enterpriseToRevenue"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
            }

                # Add valuation metrics to financial_metrics
                for metric_name, value in valuation_metrics.items():
                    financial_metrics[metric_name] = value

                # Profitability metrics with citations
                profitability_metrics = {
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "return_on_equity": info.get("returnOnEquity"),
                    "return_on_assets": info.get("returnOnAssets"),
                    "return_on_investment": info.get("returnOnInvestment"),
                }

                # Add profitability metrics to financial_metrics
                for metric_name, value in profitability_metrics.items():
                    financial_metrics[metric_name] = value

                # Financial health metrics with citations
                financial_health_metrics = {
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "total_cash": info.get("totalCash"),
                    "total_debt": info.get("totalDebt"),
                    "book_value": info.get("bookValue"),
                }

                # Add financial health metrics to financial_metrics
                for metric_name, value in financial_health_metrics.items():
                    financial_metrics[metric_name] = value

                # Additional metrics with citations
                additional_metrics = {
                    # Dividend metrics
                    "dividend_yield": info.get("dividendYield"),
                    "dividend_rate": info.get("dividendRate"),
                    "payout_ratio": info.get("payoutRatio"),
                    "ex_dividend_date": info.get("exDividendDate"),

                    # Trading metrics
                    "volume": info.get("volume"),
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10days": info.get("averageVolume10days"),
                    "beta": info.get("beta"),

                    # Analyst metrics
                    "target_high_price": info.get("targetHighPrice"),
                    "target_low_price": info.get("targetLowPrice"),
                    "target_mean_price": info.get("targetMeanPrice"),
                    "target_median_price": info.get("targetMedianPrice"),
                    "recommendation_mean": info.get("recommendationMean"),
                    "recommendation_key": info.get("recommendationKey"),
                    "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),

                    # Growth metrics
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                    "revenue_quarterly_growth": info.get("revenueQuarterlyGrowth"),

                    # Additional HK-specific metrics
                    "currency": info.get("currency"),
                    "exchange": info.get("exchange"),
                    "quote_type": info.get("quoteType"),
                    "market_state": info.get("marketState"),
                    "timezone": info.get("timeZoneFullName")
                }

                # Add additional metrics to financial_metrics
                for metric_name, value in additional_metrics.items():
                    financial_metrics[metric_name] = value

                # Track Yahoo Finance as a data source (not individual citations)
                total_metrics = len([v for v in financial_metrics.values() if v is not None])
                logger.info(f"📊 Tracking {total_metrics} metrics from Yahoo Finance for {ticker}")
                if total_metrics > 0:
                    self.citation_tracker.track_yahoo_finance_data(
                        ticker, total_metrics, f"yfinance.Ticker('{ticker}').info"
                    )
                    logger.info(f"✅ Data source tracked for {ticker}: {total_metrics} metrics")

                # Calculate data completeness score
                total_fields = len(financial_metrics)
                filled_fields = sum(1 for v in financial_metrics.values() if v is not None)
                completeness_score = (filled_fields / total_fields) * 100 if total_fields > 0 else 0

                return {
                    "financial_metrics": financial_metrics,
                    "data_quality": {
                        "completeness_score": completeness_score,
                        "filled_fields": filled_fields,
                        "total_fields": total_fields,
                        "source": "yahoo_finance",
                        "enhanced_collection": info != (await loop.run_in_executor(self.executor, self._safe_get_info, stock))
                    }
                }
            except Exception as e:
                # Re-raise for retry mechanism to handle
                raise Exception(f"Failed to get financial metrics for {ticker}: {e}")

        try:
            return await self._retry_with_backoff(_fetch_financial_metrics)
        except Exception as e:
            logger.warning(f"All attempts failed for financial metrics {ticker}: {e}")
            return {
                "financial_metrics": {},
                "data_quality": {
                    "completeness_score": 0,
                    "filled_fields": 0,
                    "total_fields": 0,
                    "source": "yahoo_finance",
                    "error": str(e)
                }
            }
    
    async def _get_historical_data(self, stock: yf.Ticker, ticker: str, time_period: str) -> Dict[str, Any]:
        """Get historical price data with retry logic and enhanced error handling."""
        async def _fetch_historical_data():
            try:
                loop = asyncio.get_event_loop()
                period = self.time_periods[time_period]

                # Use safe history retrieval with timeout
                hist = await loop.run_in_executor(
                    self.executor,
                    self._safe_get_history,
                    stock,
                    period
                )

                if hist.empty:
                    logger.warning(f"No historical data found for {ticker} - trying alternative methods")
                    hist = await self._get_alternative_historical_data(ticker, period)

                if hist.empty:
                    return {"historical_data": {}}

                # Convert to serializable format
                historical_data = {
                    "period": time_period,
                    "start_date": hist.index[0].isoformat(),
                    "end_date": hist.index[-1].isoformat(),
                    "data_points": len(hist),
                    "prices": {
                        "dates": [date.isoformat() for date in hist.index],
                        "open": hist["Open"].tolist(),
                        "high": hist["High"].tolist(),
                        "low": hist["Low"].tolist(),
                        "close": hist["Close"].tolist(),
                        "volume": hist["Volume"].tolist()
                    },
                    "summary": {
                        "current_price": float(hist["Close"].iloc[-1]),
                        "period_high": float(hist["High"].max()),
                        "period_low": float(hist["Low"].min()),
                        "period_return": float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100),
                        "volatility": float(hist["Close"].pct_change().std() * 100)
                    }
                }
            
                return {"historical_data": historical_data}

            except Exception as e:
                # Re-raise for retry mechanism to handle
                raise Exception(f"Failed to get historical data for {ticker}: {e}")

        try:
            return await self._retry_with_backoff(_fetch_historical_data)
        except Exception as e:
            logger.warning(f"All attempts failed for historical data {ticker}: {e}")
            return {"historical_data": {}}

    async def _get_historical_data_with_cache(self, stock: yf.Ticker, ticker: str, time_period: str) -> Dict[str, Any]:
        """Get historical price data with caching support."""
        try:
            # Get historical data using existing method
            result = await self._get_historical_data(stock, ticker, time_period)

            # Store in caches if data was retrieved successfully
            if result.get("historical_data"):
                metadata = {
                    "ticker": ticker,
                    "time_period": time_period,
                    "data_points": result["historical_data"].get("data_points", 0),
                    "period": result["historical_data"].get("period", time_period),
                    "api_source": "yahoo_finance"
                }

                # Primary storage: PostgreSQL cache
                if self.cache_manager and self.cache_manager.available:
                    pg_success = await self.cache_manager.store_cached_data(
                        ticker, "historical_data", result, ttl_hours=24, metadata=metadata
                    )
                    if pg_success:
                        logger.info(f"🗄️ Stored historical data in PostgreSQL for {ticker} ({metadata['data_points']} points)")
                    else:
                        logger.warning(f"⚠️ Failed to store historical data in PostgreSQL for {ticker}")

                # Secondary storage: File cache (backup)
                if self.historical_cache:
                    file_success = await self.historical_cache.store_data(ticker, result, metadata)
                    if file_success:
                        logger.info(f"📁 Stored historical data in file cache for {ticker} ({metadata['data_points']} points)")
                    else:
                        logger.warning(f"⚠️ Failed to store historical data in file cache for {ticker}")

            return result

        except Exception as e:
            logger.error(f"❌ Error in cached historical data retrieval for {ticker}: {e}")
            return {"historical_data": {}}

    async def _get_cached_historical_data(self, ticker: str, time_period: str) -> Optional[Dict[str, Any]]:
        """
        Three-tier cache retrieval: PostgreSQL -> File Cache -> None

        Args:
            ticker: Stock ticker symbol
            time_period: Time period for data

        Returns:
            Cached data if found, None otherwise
        """
        try:
            # Tier 1: Check PostgreSQL cache first
            if self.cache_manager and self.cache_manager.available:
                pg_data = await self.cache_manager._get_historical_data(ticker, time_period)
                if pg_data:
                    logger.info(f"🗄️ PostgreSQL cache hit for {ticker}:{time_period}")
                    return pg_data

            # Tier 2: Check file cache
            if self.historical_cache:
                file_data = await self.historical_cache.get_cached_data(ticker)
                if file_data:
                    logger.info(f"📁 File cache hit for {ticker}:{time_period}")

                    # If we have PostgreSQL available, store this data there for future use
                    if self.cache_manager and self.cache_manager.available:
                        # Extract the historical data from file cache structure
                        promotion_data = file_data.get('data', file_data)
                        if promotion_data:
                            await self.cache_manager.store_cached_data(
                                ticker, "historical_data", promotion_data, ttl_hours=24
                            )
                            logger.info(f"📤 Promoted file cache data to PostgreSQL for {ticker}")
                        else:
                            logger.warning(f"⚠️ No valid data to promote for {ticker}")

                    # Return file cache data as-is (will be normalized in the calling function)
                    return file_data

            # Tier 3: No cache hit
            logger.info(f"❌ No cached data found for {ticker}:{time_period}")
            return None

        except Exception as e:
            logger.error(f"❌ Error in three-tier cache retrieval for {ticker}: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.historical_cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            **self.historical_cache.get_cache_stats()
        }

    async def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up expired cache files."""
        if not self.historical_cache:
            return {"cache_enabled": False}

        return await self.historical_cache.cleanup_expired_cache()

    def _safe_get_history(self, stock: yf.Ticker, period: str) -> pd.DataFrame:
        """Safely get historical data with enhanced error handling."""
        try:
            # Add timeout to prevent hanging requests
            hist = stock.history(period=period, timeout=30)

            if hist is None or hist.empty:
                logger.warning(f"Empty historical data response for period {period}")
                return pd.DataFrame()

            return hist

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in stock.history: {e}")
            raise Exception(f"JSON parsing failed: {e}")
        except HTTPError as e:
            if e.response.status_code == 429:
                raise Exception(f"Rate limited (429): {e}")
            else:
                raise Exception(f"HTTP error {e.response.status_code}: {e}")
        except (RequestException, Timeout) as e:
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error in _safe_get_history: {e}")
            raise Exception(f"Unexpected error: {e}")

    async def _get_alternative_historical_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Get historical data using alternative methods for Hong Kong tickers."""
        try:
            # Try different ticker formats for HK stocks
            if ticker.endswith('.HK'):
                alternative_formats = [
                    ticker.replace('.HK', '.HKG'),  # Alternative format
                    f"{ticker.split('.')[0].zfill(4)}.HK",  # Ensure 4-digit format
                ]

                for alt_ticker in alternative_formats:
                    try:
                        alt_stock = yf.Ticker(alt_ticker)
                        loop = asyncio.get_event_loop()
                        alt_hist = await loop.run_in_executor(
                            self.executor,
                            self._safe_get_history,
                            alt_stock,
                            period
                        )

                        if not alt_hist.empty:
                            logger.info(f"✅ Retrieved historical data for {ticker} using format {alt_ticker}")
                            return alt_hist

                    except Exception as e:
                        logger.debug(f"Failed alternative format {alt_ticker}: {e}")
                        continue

            # Try shorter periods if the requested period fails
            fallback_periods = ['5d', '1mo', '3mo']
            if period not in fallback_periods:
                for fallback_period in fallback_periods:
                    try:
                        stock = yf.Ticker(ticker)
                        loop = asyncio.get_event_loop()
                        hist = await loop.run_in_executor(
                            self.executor,
                            self._safe_get_history,
                            stock,
                            fallback_period
                        )

                        if not hist.empty:
                            logger.info(f"✅ Retrieved historical data for {ticker} using fallback period {fallback_period}")
                            return hist

                    except Exception as e:
                        logger.debug(f"Failed fallback period {fallback_period}: {e}")
                        continue

            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"All alternative historical data methods failed for {ticker}: {e}")
            return pd.DataFrame()

    async def _get_enhanced_hk_data(self, stock: yf.Ticker, ticker: str) -> Dict[str, Any]:
        """
        Enhanced data collection for Hong Kong tickers with multiple fallback strategies.
        """
        enhanced_info = {}

        try:
            # Strategy 1: Try different ticker formats for HK stocks
            if ticker.endswith('.HK'):
                alternative_formats = [
                    ticker,  # Original format (e.g., 0005.HK)
                    ticker.replace('.HK', '.HKG'),  # Alternative format
                    f"{ticker.split('.')[0].zfill(4)}.HK",  # Ensure 4-digit format
                ]

                for alt_ticker in alternative_formats:
                    try:
                        alt_stock = yf.Ticker(alt_ticker)
                        loop = asyncio.get_event_loop()
                        alt_info = await loop.run_in_executor(self.executor, lambda: alt_stock.info)

                        if alt_info and alt_info.get('symbol'):
                            logger.info(f"Successfully retrieved data for {ticker} using format {alt_ticker}")
                            enhanced_info.update(alt_info)
                            break
                    except Exception as e:
                        logger.debug(f"Failed to get data with format {alt_ticker}: {e}")
                        continue

            # Strategy 2: Try to get basic price data from history
            if not enhanced_info.get('currentPrice'):
                try:
                    loop = asyncio.get_event_loop()
                    hist = await loop.run_in_executor(
                        self.executor,
                        lambda: stock.history(period="5d")
                    )

                    if not hist.empty:
                        latest = hist.iloc[-1]
                        enhanced_info.update({
                            'currentPrice': latest.get('Close'),
                            'previousClose': hist.iloc[-2].get('Close') if len(hist) > 1 else latest.get('Close'),
                            'volume': latest.get('Volume'),
                            'dayHigh': latest.get('High'),
                            'dayLow': latest.get('Low'),
                            'open': latest.get('Open')
                        })
                        logger.info(f"Retrieved price data from history for {ticker}")
                except Exception as e:
                    logger.debug(f"Failed to get historical price data for {ticker}: {e}")

            # Strategy 3: Estimate missing ratios from available data
            if enhanced_info.get('currentPrice') and not enhanced_info.get('marketCap'):
                try:
                    # Try to get shares outstanding from financials
                    loop = asyncio.get_event_loop()
                    financials = await loop.run_in_executor(
                        self.executor,
                        lambda: stock.get_shares_full(start="2023-01-01")
                    )

                    if financials is not None and not financials.empty:
                        shares = financials.iloc[-1] if len(financials) > 0 else None
                        if shares:
                            market_cap = enhanced_info['currentPrice'] * shares
                            enhanced_info['marketCap'] = market_cap
                            enhanced_info['sharesOutstanding'] = shares
                            logger.info(f"Calculated market cap for {ticker}: {market_cap}")
                except Exception as e:
                    logger.debug(f"Failed to calculate market cap for {ticker}: {e}")

            return enhanced_info

        except Exception as e:
            logger.warning(f"Enhanced HK data collection failed for {ticker}: {e}")
            return enhanced_info
    
    async def _get_company_info(self, stock: yf.Ticker, ticker: str) -> Dict[str, Any]:
        """Get detailed company information."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(self.executor, lambda: stock.info)

            # Handle None info for Hong Kong tickers and other cases
            if info is None:
                logger.warning(f"No company info available for {ticker} - using empty data")
                info = {}

            return {
                "company_info": {
                    "business_summary": info.get("longBusinessSummary", ""),
                    "full_time_employees": info.get("fullTimeEmployees"),
                    "city": info.get("city", ""),
                    "state": info.get("state", ""),
                    "country": info.get("country", ""),
                    "phone": info.get("phone", ""),
                    "website": info.get("website", ""),
                    "logo_url": info.get("logo_url", ""),
                    "recommendation_key": info.get("recommendationKey", ""),
                    "recommendation_mean": info.get("recommendationMean"),
                    "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                    "target_high_price": info.get("targetHighPrice"),
                    "target_low_price": info.get("targetLowPrice"),
                    "target_mean_price": info.get("targetMeanPrice"),
                    "target_median_price": info.get("targetMedianPrice")
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get company info for {ticker}: {e}")
            return {"company_info": {}}

    async def fill_missing_metrics(self, metrics: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Intelligently fill missing financial metrics using estimation and fallback methods.
        """
        filled_metrics = metrics.copy()

        try:
            # Fill missing ratios using available data
            current_price = filled_metrics.get('current_price')
            market_cap = filled_metrics.get('market_cap')

            # Estimate P/E ratio if earnings data is available
            if not filled_metrics.get('pe_ratio') and current_price:
                try:
                    stock = yf.Ticker(ticker)
                    loop = asyncio.get_event_loop()
                    financials = await loop.run_in_executor(
                        self.executor,
                        lambda: stock.financials
                    )

                    if financials is not None and not financials.empty:
                        # Get latest earnings per share
                        net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else None
                        shares = filled_metrics.get('shares_outstanding')

                        if net_income and shares and net_income > 0:
                            eps = net_income / shares
                            pe_ratio = current_price / eps
                            filled_metrics['pe_ratio'] = pe_ratio
                            filled_metrics['estimated_pe'] = True
                            logger.info(f"Estimated P/E ratio for {ticker}: {pe_ratio:.2f}")
                except Exception as e:
                    logger.debug(f"Failed to estimate P/E ratio for {ticker}: {e}")

            # Estimate market cap if shares outstanding is available
            if not market_cap and current_price and filled_metrics.get('shares_outstanding'):
                estimated_market_cap = current_price * filled_metrics['shares_outstanding']
                filled_metrics['market_cap'] = estimated_market_cap
                filled_metrics['estimated_market_cap'] = True
                logger.info(f"Estimated market cap for {ticker}: {estimated_market_cap}")

            # Add data source attribution
            filled_metrics['data_sources'] = {
                'yahoo_finance': True,
                'estimated_metrics': any(k.startswith('estimated_') for k in filled_metrics.keys()),
                'enhancement_applied': len([k for k in filled_metrics.keys() if k.startswith('estimated_')]) > 0
            }

            return filled_metrics

        except Exception as e:
            logger.warning(f"Failed to fill missing metrics for {ticker}: {e}")
            return filled_metrics

    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol is valid, with enhanced support for Hong Kong tickers (XXXX.HK).

        Strategy:
        - Accepts direct Yahoo Finance info when available
        - For HK tickers like 0700.HK, tries alternate formats (.HKG) and zero-padded variants
        - Falls back to basic format validation to avoid hard-failing legitimate HK tickers
        """
        try:
            # Quick accept for reasonable HK format: 4 digits + .HK
            if re.match(r"^\d{4}\.HK$", ticker.upper()):
                # Try yfinance first
                try:
                    info = yf.Ticker(ticker).info
                    if info and info.get("symbol"):
                        return True
                except Exception:
                    pass
                # Try alternate HK format
                for alt in [ticker.upper().replace('.HK', '.HKG'), f"{ticker.split('.')[0].zfill(4)}.HK"]:
                    try:
                        info_alt = yf.Ticker(alt).info
                        if info_alt and info_alt.get("symbol"):
                            return True
                    except Exception:
                        continue
                # If network/data is flaky, still allow HK format to proceed
                return True

            # Non-HK tickers: use yfinance validation
            info = yf.Ticker(ticker).info
            return bool(info and info.get("symbol"))
        except Exception:
            # As a last resort, accept HK format even if yfinance fails
            return bool(re.match(r"^\d{4}\.HK$", ticker.upper()))

    def get_supported_periods(self) -> List[str]:
        """Get list of supported time periods."""
        return list(self.time_periods.keys())

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
