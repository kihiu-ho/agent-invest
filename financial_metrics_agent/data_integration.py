#!/usr/bin/env python3
"""
Data Integration Manager for Financial Analysis

Handles data collection, integration, and caching for financial metrics.
Coordinates between multiple data sources with intelligent caching.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    enabled: bool = True
    cache_ttl_hours: int = 24
    priority: int = 1
    timeout_seconds: int = 60

@dataclass
class DataCollectionResult:
    """Result of data collection from a source."""
    source: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cached: bool = False
    execution_time: float = 0.0
    timestamp: Optional[datetime] = None

class FinancialDataIntegrator:
    """
    Manages data collection and integration from multiple financial data sources.
    
    Features:
    - Intelligent caching with PostgreSQL
    - Parallel data collection
    - Graceful fallback handling
    - Data validation and cleaning
    - Source prioritization
    """
    
    def __init__(self, cache_manager=None, market_data_collector=None, 
                 hk_web_scraper=None, hk_data_downloader=None):
        """
        Initialize the data integrator.
        
        Args:
            cache_manager: Cache manager instance
            market_data_collector: Market data collector instance
            hk_web_scraper: HK web scraper instance
            hk_data_downloader: HK data downloader instance
        """
        self.cache_manager = cache_manager
        self.market_data_collector = market_data_collector
        self.hk_web_scraper = hk_web_scraper
        self.hk_data_downloader = hk_data_downloader
        
        # Data source configuration
        self.data_sources = {
            "market_data": DataSource("market_data", enabled=True, priority=1),
            "web_scraping": DataSource("web_scraping", enabled=True, priority=2, cache_ttl_hours=12),
            "hk_enhanced": DataSource("hk_enhanced", enabled=True, priority=3, cache_ttl_hours=6),
            "financial_metrics": DataSource("financial_metrics", enabled=True, priority=1)
        }
        
        logger.info("✅ Financial data integrator initialized")
    
    async def collect_comprehensive_data(self, ticker: str, time_period: str = "1Y", 
                                       is_hk_ticker: bool = False) -> Dict[str, Any]:
        """
        Collect comprehensive financial data from all available sources.
        
        Args:
            ticker: Stock ticker symbol
            time_period: Time period for historical data
            is_hk_ticker: Whether this is a Hong Kong ticker
            
        Returns:
            Integrated data from all sources
        """
        logger.info(f"🔄 Starting comprehensive data collection for {ticker}")
        
        # Prepare collection tasks
        collection_tasks = []
        
        # Market data collection
        if self.data_sources["market_data"].enabled and self.market_data_collector:
            collection_tasks.append(
                self._collect_with_cache("market_data", ticker, 
                                       self._collect_market_data, ticker, time_period)
            )
        
        # Web scraping (for HK tickers)
        if is_hk_ticker and self.data_sources["web_scraping"].enabled and self.hk_web_scraper:
            collection_tasks.append(
                self._collect_with_cache("web_scraping", ticker,
                                       self._collect_web_scraping_data, ticker)
            )
        
        # Enhanced HK data (if available)
        if (is_hk_ticker and self.data_sources["hk_enhanced"].enabled and 
            self.hk_data_downloader):
            collection_tasks.append(
                self._collect_with_cache("hk_enhanced", ticker,
                                       self._collect_hk_enhanced_data, ticker)
            )
        
        # Execute all collection tasks
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Integrate results
        integrated_data = await self._integrate_collection_results(ticker, results)
        
        logger.info(f"✅ Comprehensive data collection completed for {ticker}")
        return integrated_data
    
    async def _collect_with_cache(self, data_type: str, ticker: str, 
                                collection_func, *args, **kwargs) -> DataCollectionResult:
        """
        Collect data with intelligent caching.
        
        Args:
            data_type: Type of data being collected
            ticker: Stock ticker symbol
            collection_func: Function to collect data
            *args, **kwargs: Arguments for collection function
            
        Returns:
            DataCollectionResult
        """
        start_time = asyncio.get_event_loop().time()
        
        # Check cache first
        cached_data = None
        if self.cache_manager:
            cached_data = await self.cache_manager.get_cached_data(ticker, data_type)
        
        if cached_data:
            execution_time = asyncio.get_event_loop().time() - start_time
            return DataCollectionResult(
                source=data_type,
                success=True,
                data=cached_data,
                cached=True,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
        
        # Collect fresh data
        try:
            data = await collection_func(*args, **kwargs)
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Cache the result
            if self.cache_manager and data:
                source_config = self.data_sources.get(data_type)
                ttl_hours = source_config.cache_ttl_hours if source_config else 24
                await self.cache_manager.store_cached_data(
                    ticker, data_type, data, ttl_hours
                )
            
            return DataCollectionResult(
                source=data_type,
                success=True,
                data=data,
                cached=False,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Data collection failed for {data_type}:{ticker}: {e}")
            
            return DataCollectionResult(
                source=data_type,
                success=False,
                error=str(e),
                cached=False,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _collect_market_data(self, ticker: str, time_period: str) -> Dict[str, Any]:
        """Collect market data using the market data collector."""
        if not self.market_data_collector:
            raise ValueError("Market data collector not available")
        
        result = await self.market_data_collector.collect_ticker_data(ticker, time_period)
        
        if not result.get('success', False):
            raise Exception(f"Market data collection failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def _collect_web_scraping_data(self, ticker: str) -> Dict[str, Any]:
        """Collect web scraping data for HK tickers."""
        if not self.hk_web_scraper:
            raise ValueError("HK web scraper not available")
        
        result = await self.hk_web_scraper.scrape_enhanced_comprehensive_data(ticker)
        return result
    
    async def _collect_hk_enhanced_data(self, ticker: str) -> Dict[str, Any]:
        """Collect enhanced HK data using the HK data downloader."""
        if not self.hk_data_downloader:
            raise ValueError("HK data downloader not available")
        
        try:
            result = await self.hk_data_downloader.execute_comprehensive_scraping(ticker)
            return result
        except AttributeError:
            # Method doesn't exist, return empty result
            return {"status": "not_available", "reason": "Method not implemented"}
    
    async def _integrate_collection_results(self, ticker: str, 
                                          results: List[Union[DataCollectionResult, Exception]]) -> Dict[str, Any]:
        """
        Integrate data collection results from multiple sources.
        
        Args:
            ticker: Stock ticker symbol
            results: List of collection results or exceptions
            
        Returns:
            Integrated data dictionary
        """
        integrated_data = {
            "ticker": ticker,
            "success": False,
            "data_sources": {},
            "collection_summary": {
                "total_sources": len(results),
                "successful_sources": 0,
                "failed_sources": 0,
                "cached_sources": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        successful_data = {}
        
        for result in results:
            if isinstance(result, Exception):
                integrated_data["collection_summary"]["failed_sources"] += 1
                logger.error(f"❌ Collection task failed: {result}")
                continue
            
            if isinstance(result, DataCollectionResult):
                # Record collection metadata
                integrated_data["data_sources"][result.source] = {
                    "success": result.success,
                    "cached": result.cached,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                    "error": result.error
                }
                
                if result.success and result.data:
                    successful_data[result.source] = result.data
                    integrated_data["collection_summary"]["successful_sources"] += 1
                    
                    if result.cached:
                        integrated_data["collection_summary"]["cached_sources"] += 1
                else:
                    integrated_data["collection_summary"]["failed_sources"] += 1
        
        # Merge successful data
        if successful_data:
            integrated_data["success"] = True
            
            # Merge market data as base
            if "market_data" in successful_data:
                base_data = successful_data["market_data"]
                integrated_data.update(base_data)
            
            # Add web scraping data
            if "web_scraping" in successful_data:
                integrated_data["web_scraping"] = successful_data["web_scraping"]
            
            # Add enhanced HK data
            if "hk_enhanced" in successful_data:
                integrated_data["hk_enhanced"] = successful_data["hk_enhanced"]
            
            # Apply intelligent gap filling if market data exists
            if "market_data" in successful_data and self.market_data_collector:
                try:
                    if integrated_data.get('financial_metrics'):
                        enhanced_metrics = await self.market_data_collector.fill_missing_metrics(
                            integrated_data['financial_metrics'], ticker
                        )
                        integrated_data['financial_metrics'] = enhanced_metrics
                        logger.info(f"✅ Applied intelligent gap filling for {ticker}")
                except Exception as e:
                    logger.warning(f"⚠️ Gap filling failed for {ticker}: {e}")
        
        return integrated_data
    
    async def invalidate_ticker_cache(self, ticker: str, data_types: Optional[List[str]] = None):
        """
        Invalidate cached data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data_types: Specific data types to invalidate (None for all)
        """
        if not self.cache_manager:
            return
        
        if data_types:
            for data_type in data_types:
                await self.cache_manager.invalidate_cache(ticker, data_type)
        else:
            await self.cache_manager.invalidate_cache(ticker)
        
        logger.info(f"✅ Invalidated cache for {ticker}")
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        return {
            "sources": {name: {
                "enabled": source.enabled,
                "priority": source.priority,
                "cache_ttl_hours": source.cache_ttl_hours,
                "available": self._is_source_available(name)
            } for name, source in self.data_sources.items()},
            "cache_available": self.cache_manager is not None and self.cache_manager.available
        }
    
    def _is_source_available(self, source_name: str) -> bool:
        """Check if a data source is available."""
        source_map = {
            "market_data": self.market_data_collector,
            "web_scraping": self.hk_web_scraper,
            "hk_enhanced": self.hk_data_downloader,
            "financial_metrics": self.market_data_collector
        }
        
        return source_map.get(source_name) is not None
