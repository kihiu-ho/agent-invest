#!/usr/bin/env python3
"""
Simplified Hong Kong Stock Web Scraper with Markdown-Based Content Extraction.

This module provides comprehensive web scraping capabilities for Hong Kong stock data
from multiple sources including StockAnalysis.com and TipRanks.com. It uses a simplified
approach focused on extracting clean markdown content rather than structured JSON data.

Key Features:
- Clean markdown content extraction using Crawl4AI
- Multi-page data collection from multiple sources
- Simplified data structure with raw content + metadata
- Robust error handling and logging
- Support for any Hong Kong ticker symbol
- Backward compatibility with existing workflows

Author: Financial Metrics Agent System
Version: 3.0 - Simplified Markdown-Based Approach
"""

import asyncio
import logging
import os
import signal
import time
from typing import Dict, Any, Optional, List
import sys

# Check for Crawl4AI dependency with enhanced imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMConfig, DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import LLMContentFilter
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None
    LLMConfig = None
    DefaultMarkdownGenerator = None
    LLMContentFilter = None

# Import database manager and extraction constants
from database_manager import db_manager
from extraction_constants import WebScrapingMethods, ExtractionMethodValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HKStockWebScraper:
    """
    Simplified Hong Kong Stock Web Scraper using markdown-based content extraction.
    
    This scraper focuses on extracting clean markdown content from financial websites
    rather than attempting structured JSON parsing, making it more reliable and maintainable.
    """
    
    def __init__(self, timeout: int = 45, request_delay: float = 2.0, max_retries: int = 2, db_manager_instance=None):
        """
        Initialize the Hong Kong Stock Web Scraper.

        Args:
            timeout: Request timeout in seconds (increased to 45s for stability)
            request_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.timeout = timeout
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.crawl4ai_available = CRAWL4AI_AVAILABLE
        
        # Base URLs for data sources
        self.stockanalysis_base = "https://stockanalysis.com/quote/hkg/"
        self.tipranks_base = "https://www.tipranks.com/stocks/hk:"
        
        # Enhanced URL structures for comprehensive data collection
        self.stockanalysis_urls = {
            'overview': '',  # Base URL
            'financials': 'financials/',
            'statistics': 'statistics/',
            'dividend': 'dividend/',
            'company': 'company/'
        }
        
        self.tipranks_urls = {
            'earnings': 'earnings',
            'forecast': 'forecast',
            'financials': 'financials',
            'technical': 'technical-analysis',
            'news': 'stock-news'
        }
        
        # Website configurations for extensible scraping
        self.website_configs = {
            'stockanalysis': {
                'name': 'StockAnalysis.com',
                'base_url': self.stockanalysis_base,
                'url_pattern': '{base_url}{ticker_number}/{page_suffix}',
                'page_types': self.stockanalysis_urls,
                'display_emoji': '📊'
            },
            'tipranks': {
                'name': 'TipRanks.com',
                'base_url': self.tipranks_base,
                'url_pattern': '{base_url}{ticker_number}/{page_suffix}',
                'page_types': self.tipranks_urls,
                'display_emoji': '🎯'
            }
        }

        # Initialize database manager
        self.db_manager = db_manager_instance if db_manager_instance is not None else db_manager
        self.db_initialized = False

        logger.info(f"HKStockWebScraper initialized (Crawl4AI: {'✅' if self.crawl4ai_available else '❌'})")
        logger.info(f"📋 Configured websites: {', '.join(self.website_configs.keys())}")
        logger.info(f"💾 Database caching: {'✅' if self.db_manager.available else '❌'}")

    def extract_ticker_number(self, ticker: str) -> str:
        """
        Extract the numeric part from Hong Kong ticker symbol.
        
        Args:
            ticker: Hong Kong ticker (e.g., '0700.HK')
            
        Returns:
            Numeric ticker part (e.g., '0700')
        """
        if '.HK' in ticker:
            return ticker.replace('.HK', '')
        return ticker

    async def _ensure_db_initialized(self):
        """Ensure database is initialized (called once per session)."""
        if not self.db_initialized and self.db_manager.available:
            try:
                await self.db_manager.initialize()
                self.db_initialized = True
                logger.info("✅ Database caching initialized")
            except Exception as e:
                logger.warning(f"⚠️  Database initialization failed: {e}")
                self.db_manager.available = False

    async def close(self):
        """Close the scraper and clean up database resources."""
        if self.db_manager:
            await self.db_manager.close()
            logger.info("✅ HKStockWebScraper closed")

    def _create_financial_content_filter(self) -> Optional[Any]:
        """
        Create an LLM content filter specifically for financial data extraction.

        Returns:
            LLMContentFilter instance or None if not available
        """
        if not CRAWL4AI_AVAILABLE or not LLMContentFilter or not LLMConfig:
            return None

        try:
            # Get LLM configuration from environment
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("CRAWL4AI_OPENAI_BASE_URL")
            model = os.getenv("CRAWL4AI_OPENAI_MODEL", "gpt-4o")

            if not api_key:
                logger.warning("⚠️  No OpenAI API key found - using basic markdown extraction")
                return None

            # Create LLM config
            if base_url:
                llm_config = LLMConfig(
                    provider=f"openai/{model}",
                    api_token=api_key,
                    base_url=base_url
                )
            else:
                llm_config = LLMConfig(
                    provider=f"openai/{model}",
                    api_token=api_key
                )

            # Create content filter with financial focus
            filter_instruction = """
            Focus on extracting financial and investment-related content.
            Include:
            - Stock prices, market cap, and valuation metrics
            - Financial ratios (P/E, P/B, ROE, etc.)
            - Revenue, earnings, and growth data
            - Analyst ratings and price targets
            - Dividend information and yield data
            - Company financial statements data
            - Technical analysis indicators
            - News headlines and financial updates
            - Key financial numbers and percentages

            Exclude:
            - Navigation menus and sidebars
            - Advertisement content
            - Footer and header elements
            - Social media widgets
            - Cookie notices and popups

            Preserve all financial numbers, percentages, and currency values.
            Format as clean markdown with proper headers and structure.
            Keep links to important financial documents but remove general navigation links.
            """

            return LLMContentFilter(
                llm_config=llm_config,
                instruction=filter_instruction,
                chunk_token_threshold=4096,
                verbose=True
            )

        except Exception as e:
            logger.warning(f"⚠️  Failed to create LLM content filter: {e}")
            return None

    async def scrape_website_enhanced(self, ticker: str, website_key: str) -> Dict[str, Any]:
        """
        Generalized enhanced scraping method for any configured website.

        Args:
            ticker: Hong Kong ticker symbol
            website_key: Key for website configuration (e.g., 'stockanalysis', 'tipranks')

        Returns:
            Dictionary containing markdown content from all pages of the specified website
        """
        if website_key not in self.website_configs:
            return {
                "success": False,
                "error": f"Unknown website configuration: {website_key}",
                "website": website_key,
                "ticker": ticker
            }

        config = self.website_configs[website_key]
        logger.info(f"{config['display_emoji']} Starting enhanced {config['name']} scraping for {ticker}")

        ticker_number = self.extract_ticker_number(ticker)
        results = {}

        # Scrape all pages for this website
        for page_type, page_suffix in config['page_types'].items():
            try:
                # Build URL using the website's URL pattern
                url = config['url_pattern'].format(
                    base_url=config['base_url'],
                    ticker_number=ticker_number,
                    page_suffix=page_suffix
                )

                logger.info(f"🌐 Scraping {config['name']} {page_type}: {url}")

                result = await self.scrape_page_content(url, page_type, website_key, ticker)
                results[page_type] = result

                # Add delay between requests
                await asyncio.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"❌ {config['name']} {page_type} scraping failed for {ticker}: {e}")
                results[page_type] = {
                    "success": False,
                    "error": str(e),
                    "url": url if 'url' in locals() else "unknown",
                    "page_type": page_type,
                    "source": website_key,
                    "ticker": ticker,
                    "scraped_at": time.time()
                }

        return results

    async def _perform_crawling(self, url: str, extraction_method: str):
        """
        Perform the actual crawling with proper resource management.

        Args:
            url: URL to crawl
            extraction_method: Method to use for extraction

        Returns:
            Crawling result object
        """
        crawler = None
        try:
            # Add delay between requests to be respectful
            await asyncio.sleep(self.request_delay)

            if extraction_method == WebScrapingMethods.LLM_FILTERED and self.crawl4ai_available:
                # Enhanced extraction with LLM filtering for financial content
                content_filter = self._create_financial_content_filter()

                if content_filter and DefaultMarkdownGenerator and CrawlerRunConfig:
                    md_generator = DefaultMarkdownGenerator(
                        content_filter=content_filter,
                        options={"ignore_links": False}  # Keep financial document links
                    )
                    config = CrawlerRunConfig(markdown_generator=md_generator)

                    crawler = AsyncWebCrawler(verbose=False)
                    await crawler.__aenter__()
                    result = await crawler.arun(url, config=config)
                else:
                    # Fallback to basic extraction
                    crawler = AsyncWebCrawler(verbose=False)
                    await crawler.__aenter__()
                    result = await crawler.arun(
                        url=url,
                        bypass_cache=True,
                        timeout=self.timeout
                    )
            else:
                # Basic markdown extraction
                crawler = AsyncWebCrawler(verbose=False)
                await crawler.__aenter__()
                result = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    timeout=self.timeout
                )

            return result

        except asyncio.CancelledError:
            logger.warning(f"🚫 Crawling operation cancelled for {url}")
            raise
        except Exception as e:
            logger.error(f"❌ Crawling error for {url}: {e}")
            raise
        finally:
            # Ensure proper cleanup of crawler resources
            if crawler:
                try:
                    await crawler.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Error during crawler cleanup: {cleanup_error}")

    async def scrape_page_content(self, url: str, page_type: str, source: str, ticker: str) -> Dict[str, Any]:
        """
        Scrape a single page and return markdown content with metadata.
        Uses database caching to avoid redundant crawling.

        Args:
            url: Target URL to scrape
            page_type: Type of page (overview, financials, etc.)
            source: Source name (stockanalysis, tipranks)
            ticker: Stock ticker

        Returns:
            Dictionary containing markdown content and metadata
        """
        # Initialize database if needed
        await self._ensure_db_initialized()

        # Check cache first
        cached_content = await self.db_manager.get_cached_content(ticker, url, page_type, source)
        if cached_content:
            logger.info(f"📦 Using cached content for {source} {page_type} - {ticker}")
            return cached_content

        if not self.crawl4ai_available:
            return {
                "success": False,
                "error": "Crawl4AI not available",
                "url": url,
                "page_type": page_type,
                "source": source,
                "ticker": ticker,
                "scraped_at": time.time()
            }
        
        # Retry logic for failed requests
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🌐 Scraping {source} {page_type}: {url} (attempt {attempt + 1})")

                # Determine extraction method based on availability
                extraction_method = WebScrapingMethods.get_preferred_method(self.crawl4ai_available)

                # Use asyncio.wait_for to implement timeout with proper cancellation handling
                result = await asyncio.wait_for(
                    self._perform_crawling(url, extraction_method),
                    timeout=self.timeout
                )

                if result and result.success:
                    # Extract markdown content (filtered if available)
                    if hasattr(result, 'fit_markdown') and result.fit_markdown:
                        markdown_content = result.fit_markdown
                        logger.info(f"✅ Using LLM-filtered markdown for {source} {page_type}")
                    else:
                        markdown_content = result.markdown or ""

                    html_content = result.html or ""

                    # Calculate content quality metrics
                    content_length = len(markdown_content)
                    has_meaningful_content = content_length > 100

                    # Prepare result data
                    result_data = {
                        "success": True,
                        "url": url,
                        "page_type": page_type,
                        "source": source,
                        "ticker": ticker,
                        "markdown_content": markdown_content,
                        "html_content": html_content,
                        "content_length": content_length,
                        "has_meaningful_content": has_meaningful_content,
                        "scraped_at": time.time(),
                        "extraction_method": extraction_method,
                        "attempt": attempt + 1
                    }

                    # Store in database cache (async, don't wait for completion)
                    if has_meaningful_content:
                        storage_task = self.db_manager.create_storage_task(result_data)
                        if storage_task:
                            logger.debug(f"💾 Caching content for {source} {page_type} - {ticker}")
                        else:
                            logger.debug(f"🛑 Skipping cache storage - database shutting down")

                    return result_data
                else:
                    error_msg = result.error_message if result else "Unknown crawling error"
                    if attempt < self.max_retries:
                        logger.warning(f"⚠️ Attempt {attempt + 1} failed for {source} {page_type} - {ticker}: {error_msg}. Retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Failed after {self.max_retries + 1} attempts: {error_msg}",
                            "url": url,
                            "page_type": page_type,
                            "source": source,
                            "ticker": ticker,
                            "scraped_at": time.time(),
                            "extraction_method": extraction_method,
                            "attempts": attempt + 1
                        }

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    logger.warning(f"⏰ Timeout on attempt {attempt + 1} for {source} {page_type} - {ticker}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Timeout after {self.max_retries + 1} attempts for {source} {page_type} - {ticker}")
                    return {
                        "success": False,
                        "error": f"Timeout after {self.timeout}s (attempted {self.max_retries + 1} times)",
                        "url": url,
                        "page_type": page_type,
                        "source": source,
                        "ticker": ticker,
                        "scraped_at": time.time(),
                        "attempts": attempt + 1
                    }

            except asyncio.CancelledError:
                logger.warning(f"🚫 Scraping cancelled for {source} {page_type} - {ticker}")
                return {
                    "success": False,
                    "error": "Operation cancelled",
                    "url": url,
                    "page_type": page_type,
                    "source": source,
                    "ticker": ticker,
                    "scraped_at": time.time(),
                    "attempts": attempt + 1
                }

            except KeyboardInterrupt:
                logger.warning(f"⌨️ Keyboard interrupt during scraping {source} {page_type} - {ticker}")
                return {
                    "success": False,
                    "error": "Keyboard interrupt",
                    "url": url,
                    "page_type": page_type,
                    "source": source,
                    "ticker": ticker,
                    "scraped_at": time.time(),
                    "attempts": attempt + 1
                }

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"⚠️ Exception on attempt {attempt + 1} for {source} {page_type} - {ticker}: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Failed to scrape {source} {page_type} for {ticker} after {self.max_retries + 1} attempts: {e}")
                    return {
                        "success": False,
                        "error": f"Exception after {self.max_retries + 1} attempts: {str(e)}",
                        "url": url,
                        "page_type": page_type,
                        "source": source,
                        "ticker": ticker,
                        "scraped_at": time.time(),
                        "attempts": attempt + 1
                    }
    
    async def scrape_stockanalysis_enhanced(self, ticker: str) -> Dict[str, Any]:
        """
        Scrape all StockAnalysis.com pages for a Hong Kong ticker.
        
        Args:
            ticker: Hong Kong ticker symbol
            
        Returns:
            Dictionary containing markdown content from all StockAnalysis pages
        """
        return await self.scrape_website_enhanced(ticker, 'stockanalysis')
    
    async def scrape_tipranks_enhanced(self, ticker: str) -> Dict[str, Any]:
        """
        Scrape all TipRanks.com pages for a Hong Kong ticker.
        
        Args:
            ticker: Hong Kong ticker symbol
            
        Returns:
            Dictionary containing markdown content from all TipRanks pages
        """
        return await self.scrape_website_enhanced(ticker, 'tipranks')

    async def scrape_stockanalysis_data(self, ticker: str) -> Dict[str, Any]:
        """
        Scrape basic StockAnalysis.com data (backward compatibility method).

        Args:
            ticker: Hong Kong ticker symbol

        Returns:
            Dictionary containing overview page markdown content
        """
        ticker_number = self.extract_ticker_number(ticker)
        url = f"{self.stockanalysis_base}{ticker_number}/"

        result = await self.scrape_page_content(url, "overview", "stockanalysis", ticker)
        return result

    async def scrape_tipranks_data(self, ticker: str) -> Dict[str, Any]:
        """
        Scrape basic TipRanks.com data (backward compatibility method).

        Args:
            ticker: Hong Kong ticker symbol

        Returns:
            Dictionary containing earnings page markdown content
        """
        ticker_number = self.extract_ticker_number(ticker)
        url = f"{self.tipranks_base}{ticker_number}/earnings"

        result = await self.scrape_page_content(url, "earnings", "tipranks", ticker)
        return result

    async def scrape_comprehensive_data(self, ticker: str, include_enhanced: bool = True) -> Dict[str, Any]:
        """
        Scrape comprehensive data from all available sources.

        Args:
            ticker: Hong Kong ticker symbol (e.g., '0700.HK')
            include_enhanced: Whether to include enhanced multi-page scraping

        Returns:
            Dictionary containing all scraped markdown content
        """
        logger.info(f"🌐 Starting comprehensive scraping for {ticker} (Enhanced: {include_enhanced})")

        start_time = time.time()
        results = {}

        if include_enhanced:
            # Enhanced comprehensive scraping with all pages
            results = await self.scrape_enhanced_comprehensive_data(ticker)
        else:
            # Basic scraping (backward compatibility)
            results = await self._scrape_basic_comprehensive_data(ticker)

        # Calculate summary statistics
        total_time = time.time() - start_time

        # Count all attempted and successful sources and determine extraction method used
        sources_attempted = 0
        sources_successful = 0
        total_content_length = 0
        extraction_methods_used = set()

        for key, value in results.items():
            if key != 'scraping_summary' and isinstance(value, dict):
                if 'success' in value:
                    sources_attempted += 1
                    if value.get('success'):
                        sources_successful += 1
                        total_content_length += value.get('content_length', 0)
                    # Track extraction method used
                    if 'extraction_method' in value:
                        extraction_methods_used.add(value['extraction_method'])
                elif isinstance(value, dict):
                    # Handle nested results (like stockanalysis_enhanced)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict) and 'success' in sub_value:
                            sources_attempted += 1
                            if sub_value.get('success'):
                                sources_successful += 1
                                total_content_length += sub_value.get('content_length', 0)
                            # Track extraction method used
                            if 'extraction_method' in sub_value:
                                extraction_methods_used.add(sub_value['extraction_method'])

        # Determine the primary extraction method used
        if extraction_methods_used:
            # Prefer llm_filtered if it was used, otherwise use the first method found
            if WebScrapingMethods.LLM_FILTERED in extraction_methods_used:
                primary_extraction_method = WebScrapingMethods.LLM_FILTERED
            else:
                primary_extraction_method = next(iter(extraction_methods_used))
        else:
            # Fallback to the preferred method based on availability
            primary_extraction_method = WebScrapingMethods.get_preferred_method(self.crawl4ai_available)

        results['scraping_summary'] = {
            'total_time': total_time,
            'sources_attempted': sources_attempted,
            'sources_successful': sources_successful,
            'success_rate': sources_successful / sources_attempted if sources_attempted > 0 else 0,
            'total_content_length': total_content_length,
            'ticker': ticker,
            'enhanced_mode': include_enhanced,
            'extraction_method': primary_extraction_method,
            'extraction_methods_used': list(extraction_methods_used) if extraction_methods_used else []
        }

        logger.info(f"✅ Comprehensive scraping completed for {ticker}: {sources_successful}/{sources_attempted} sources successful in {total_time:.2f}s")

        return results

    async def _scrape_basic_comprehensive_data(self, ticker: str) -> Dict[str, Any]:
        """Basic scraping for backward compatibility."""
        results = {}

        # Scrape StockAnalysis data
        try:
            stockanalysis_result = await self.scrape_stockanalysis_data(ticker)
            results['stockanalysis'] = stockanalysis_result
        except Exception as e:
            logger.error(f"❌ StockAnalysis scraping failed for {ticker}: {e}")
            results['stockanalysis'] = {"success": False, "error": str(e)}

        # Add delay between requests
        await asyncio.sleep(self.request_delay)

        # Scrape TipRanks data
        try:
            tipranks_result = await self.scrape_tipranks_data(ticker)
            results['tipranks'] = tipranks_result
        except Exception as e:
            logger.error(f"❌ TipRanks scraping failed for {ticker}: {e}")
            results['tipranks'] = {"success": False, "error": str(e)}

        return results

    async def scrape_enhanced_comprehensive_data(self, ticker: str, overall_timeout: int = 300) -> Dict[str, Any]:
        """
        Enhanced comprehensive scraping with multi-page data collection and timeout protection.

        Args:
            ticker: Hong Kong ticker symbol (e.g., '0700.HK')
            overall_timeout: Overall timeout for the entire scraping operation (default: 5 minutes)

        Returns:
            Dictionary containing all enhanced scraped markdown content
        """
        logger.info(f"🚀 Starting enhanced comprehensive scraping for {ticker}")
        start_time = time.time()

        try:
            # Use asyncio.wait_for to implement overall timeout
            results = await asyncio.wait_for(
                self._perform_enhanced_comprehensive_scraping(ticker),
                timeout=overall_timeout
            )

            total_time = time.time() - start_time
            logger.info(f"✅ Enhanced comprehensive scraping completed for {ticker} in {total_time:.2f}s")
            return results

        except asyncio.TimeoutError:
            total_time = time.time() - start_time
            logger.error(f"❌ Comprehensive scraping timeout after {total_time:.2f}s for {ticker}")
            return {
                "success": False,
                "error": f"Overall operation timeout after {overall_timeout}s",
                "ticker": ticker,
                "processing_time": total_time,
                "scraped_at": time.time()
            }

        except asyncio.CancelledError:
            total_time = time.time() - start_time
            logger.warning(f"🚫 Comprehensive scraping cancelled for {ticker} after {total_time:.2f}s")
            return {
                "success": False,
                "error": "Operation cancelled",
                "ticker": ticker,
                "processing_time": total_time,
                "scraped_at": time.time()
            }

        except KeyboardInterrupt:
            total_time = time.time() - start_time
            logger.warning(f"⌨️ Keyboard interrupt during comprehensive scraping for {ticker} after {total_time:.2f}s")
            return {
                "success": False,
                "error": "Keyboard interrupt",
                "ticker": ticker,
                "processing_time": total_time,
                "scraped_at": time.time()
            }

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Comprehensive scraping failed for {ticker} after {total_time:.2f}s: {e}")
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
                "processing_time": total_time,
                "scraped_at": time.time()
            }

    async def _perform_enhanced_comprehensive_scraping(self, ticker: str) -> Dict[str, Any]:
        """
        Internal method to perform the actual enhanced comprehensive scraping.

        Args:
            ticker: Hong Kong ticker symbol

        Returns:
            Dictionary containing all scraped content
        """
        results = {}

        # Enhanced StockAnalysis scraping (multiple pages)
        try:
            stockanalysis_enhanced = await self.scrape_stockanalysis_enhanced(ticker)
            results['stockanalysis_enhanced'] = stockanalysis_enhanced

            # Also include basic StockAnalysis for backward compatibility
            if stockanalysis_enhanced.get('overview', {}).get('success'):
                results['stockanalysis'] = stockanalysis_enhanced['overview']
            else:
                # Fallback to basic scraping
                basic_sa = await self.scrape_stockanalysis_data(ticker)
                results['stockanalysis'] = basic_sa

        except Exception as e:
            logger.error(f"❌ Enhanced StockAnalysis scraping failed for {ticker}: {e}")
            results['stockanalysis_enhanced'] = {"success": False, "error": str(e)}
            # Fallback to basic scraping
            try:
                basic_sa = await self.scrape_stockanalysis_data(ticker)
                results['stockanalysis'] = basic_sa
            except Exception as e2:
                results['stockanalysis'] = {"success": False, "error": str(e2)}

        # Enhanced TipRanks scraping (multiple pages)
        try:
            tipranks_enhanced = await self.scrape_tipranks_enhanced(ticker)
            results['tipranks_enhanced'] = tipranks_enhanced

            # Also include basic TipRanks for backward compatibility
            if tipranks_enhanced.get('earnings', {}).get('success'):
                results['tipranks'] = tipranks_enhanced['earnings']
            else:
                # Fallback to basic scraping
                basic_tr = await self.scrape_tipranks_data(ticker)
                results['tipranks'] = basic_tr

        except Exception as e:
            logger.error(f"❌ Enhanced TipRanks scraping failed for {ticker}: {e}")
            results['tipranks_enhanced'] = {"success": False, "error": str(e)}
            # Fallback to basic scraping
            try:
                basic_tr = await self.scrape_tipranks_data(ticker)
                results['tipranks'] = basic_tr
            except Exception as e2:
                results['tipranks'] = {"success": False, "error": str(e2)}

        return results

    def validate_scraped_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate scraped markdown content and provide quality metrics.

        Args:
            data: Scraped data dictionary

        Returns:
            Validation results with quality metrics
        """
        validation_results = {
            'is_valid': False,
            'content_quality': 'poor',
            'issues': [],
            'metrics': {}
        }

        try:
            if not data.get('success'):
                validation_results['issues'].append('Scraping failed')
                return validation_results

            markdown_content = data.get('markdown_content', '')
            content_length = len(markdown_content)

            # Basic content validation
            if content_length < 100:
                validation_results['issues'].append('Content too short')

            if not markdown_content.strip():
                validation_results['issues'].append('Empty content')

            # Content quality assessment
            if content_length > 5000:
                validation_results['content_quality'] = 'excellent'
            elif content_length > 2000:
                validation_results['content_quality'] = 'good'
            elif content_length > 500:
                validation_results['content_quality'] = 'fair'
            else:
                validation_results['content_quality'] = 'poor'

            # Calculate metrics
            validation_results['metrics'] = {
                'content_length': content_length,
                'has_meaningful_content': data.get('has_meaningful_content', False),
                'extraction_method': data.get('extraction_method', WebScrapingMethods.UNKNOWN),
                'source': data.get('source', 'unknown'),
                'page_type': data.get('page_type', 'unknown')
            }

            # Mark as valid if no critical issues
            validation_results['is_valid'] = len(validation_results['issues']) == 0

        except Exception as e:
            validation_results['issues'].append(f'Validation error: {str(e)}')

        return validation_results
