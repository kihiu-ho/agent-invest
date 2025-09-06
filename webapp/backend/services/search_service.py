"""
Brave Search API Integration Service for AgentInvest
Handles financial data retrieval and ticker search with caching
"""

import logging
import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)

class SearchService:
    """Brave Search API integration with rate limiting and caching"""
    
    def __init__(self, cache_service=None, database_service=None):
        self.api_key = os.getenv('BRAVE_SEARCH_API_KEY', '')
        self.base_url = 'https://api.search.brave.com/res/v1'
        self.cache_service = cache_service
        self.database_service = database_service
        
        # Rate limiting configuration
        self.rate_limit_per_hour = int(os.getenv('BRAVE_SEARCH_RATE_LIMIT', '1000'))
        self.rate_limit_per_minute = int(os.getenv('BRAVE_SEARCH_RATE_LIMIT_MINUTE', '50'))
        
        # Request tracking for rate limiting
        self._request_timestamps = []
        self._session = None
        
        # Hong Kong stock market specific configuration
        self.hk_stock_patterns = [
            r'^\d{4}\.HK$',  # Standard HK stock format (e.g., 0700.HK)
            r'^\d{4}$',      # Just the number (e.g., 0700)
        ]
        
        # Financial data endpoints
        self.endpoints = {
            'web_search': '/web/search',
            'news_search': '/news/search',
            'images_search': '/images/search'
        }
    
    async def initialize(self):
        """Initialize HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip',
                    'X-Subscription-Token': self.api_key
                }
            )
            logger.info("Initialized Brave Search API session")
    
    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("Closed Brave Search API session")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Clean old timestamps (older than 1 hour)
        hour_ago = now - timedelta(hours=1)
        minute_ago = now - timedelta(minutes=1)
        
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > hour_ago
        ]
        
        # Check hourly limit
        if len(self._request_timestamps) >= self.rate_limit_per_hour:
            logger.warning("Hourly rate limit reached for Brave Search API")
            return False
        
        # Check minute limit
        recent_requests = [ts for ts in self._request_timestamps if ts > minute_ago]
        if len(recent_requests) >= self.rate_limit_per_minute:
            logger.warning("Per-minute rate limit reached for Brave Search API")
            return False
        
        return True
    
    def _record_request(self):
        """Record API request timestamp"""
        self._request_timestamps.append(datetime.utcnow())
    
    async def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if not self.api_key:
            logger.error("Brave Search API key not configured")
            return None
        
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, skipping request")
            return None
        
        await self.initialize()
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            async with self._session.get(url, params=params) as response:
                self._record_request()
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successful API request to {endpoint}")
                    return data
                elif response.status == 429:
                    logger.warning("API rate limit exceeded")
                    return None
                else:
                    logger.error(f"API request failed with status {response.status}")
                    error_text = await response.text()
                    logger.error(f"Error response: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            return None
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def _generate_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for API request"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"brave_search:{endpoint}:{param_hash}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached API response"""
        if self.cache_service:
            try:
                cached_data = await self.cache_service.get_cached_api_response(
                    'brave_search', {'cache_key': cache_key}
                )
                if cached_data:
                    logger.info(f"Cache hit for search request")
                    return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
            except Exception as e:
                logger.error(f"Failed to get cached response: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response_data: Dict) -> bool:
        """Cache API response"""
        if self.cache_service:
            try:
                return await self.cache_service.cache_api_response(
                    'brave_search',
                    {'cache_key': cache_key},
                    response_data
                )
            except Exception as e:
                logger.error(f"Failed to cache response: {e}")
        
        return False
    
    async def search_ticker_info(self, ticker_symbol: str) -> Optional[Dict]:
        """Search for Hong Kong stock ticker information"""
        # Normalize ticker symbol
        if not ticker_symbol.endswith('.HK'):
            if ticker_symbol.isdigit() and len(ticker_symbol) == 4:
                ticker_symbol = f"{ticker_symbol}.HK"
            else:
                logger.warning(f"Invalid ticker format: {ticker_symbol}")
                return None
        
        # Prepare search query
        query = f"{ticker_symbol} Hong Kong stock market financial information"
        params = {
            'q': query,
            'count': 10,
            'offset': 0,
            'mkt': 'en-US',
            'safesearch': 'moderate',
            'freshness': 'pw'  # Past week for recent information
        }
        
        # Check cache first
        cache_key = self._generate_cache_key('ticker_search', params)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Make API request
        response = await self._make_request(self.endpoints['web_search'], params)
        if response:
            # Process and filter results for financial relevance
            processed_results = self._process_ticker_results(response, ticker_symbol)
            
            # Cache the response
            await self._cache_response(cache_key, processed_results)
            
            # Store in database for analytics
            if self.database_service:
                await self._store_search_history(
                    query, 'ticker_search', len(processed_results.get('results', []))
                )
            
            return processed_results
        
        return None
    
    async def search_financial_news(self, ticker_symbol: str, 
                                  days_back: int = 7) -> Optional[Dict]:
        """Search for recent financial news about a ticker"""
        query = f"{ticker_symbol} Hong Kong stock financial news earnings"
        
        # Calculate freshness parameter
        if days_back <= 1:
            freshness = 'pd'  # Past day
        elif days_back <= 7:
            freshness = 'pw'  # Past week
        elif days_back <= 30:
            freshness = 'pm'  # Past month
        else:
            freshness = 'py'  # Past year
        
        params = {
            'q': query,
            'count': 20,
            'offset': 0,
            'mkt': 'en-US',
            'safesearch': 'moderate',
            'freshness': freshness
        }
        
        # Check cache first
        cache_key = self._generate_cache_key('news_search', params)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Make API request
        response = await self._make_request(self.endpoints['news_search'], params)
        if response:
            # Process news results
            processed_results = self._process_news_results(response, ticker_symbol)
            
            # Cache with shorter TTL for news (30 minutes)
            await self._cache_response(cache_key, processed_results)
            
            return processed_results
        
        return None
    
    async def search_company_info(self, company_name: str) -> Optional[Dict]:
        """Search for general company information"""
        query = f"{company_name} Hong Kong company profile business overview"
        params = {
            'q': query,
            'count': 15,
            'offset': 0,
            'mkt': 'en-US',
            'safesearch': 'moderate'
        }
        
        # Check cache first
        cache_key = self._generate_cache_key('company_search', params)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Make API request
        response = await self._make_request(self.endpoints['web_search'], params)
        if response:
            # Process company results
            processed_results = self._process_company_results(response, company_name)
            
            # Cache the response
            await self._cache_response(cache_key, processed_results)
            
            return processed_results
        
        return None
    
    def _process_ticker_results(self, response: Dict, ticker_symbol: str) -> Dict:
        """Process and filter ticker search results"""
        processed = {
            'ticker_symbol': ticker_symbol,
            'search_timestamp': datetime.utcnow().isoformat(),
            'results': [],
            'total_results': 0
        }
        
        if 'web' in response and 'results' in response['web']:
            results = response['web']['results']
            
            for result in results:
                # Filter for financial relevance
                title = result.get('title', '').lower()
                description = result.get('description', '').lower()
                
                # Check for financial keywords
                financial_keywords = [
                    'stock', 'share', 'financial', 'earnings', 'revenue',
                    'market', 'trading', 'investment', 'dividend', 'price'
                ]
                
                if any(keyword in title or keyword in description for keyword in financial_keywords):
                    processed_result = {
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'description': result.get('description', ''),
                        'published': result.get('age', ''),
                        'relevance_score': self._calculate_relevance_score(result, ticker_symbol)
                    }
                    processed['results'].append(processed_result)
            
            # Sort by relevance score
            processed['results'].sort(key=lambda x: x['relevance_score'], reverse=True)
            processed['total_results'] = len(processed['results'])
        
        return processed
    
    def _process_news_results(self, response: Dict, ticker_symbol: str) -> Dict:
        """Process news search results"""
        processed = {
            'ticker_symbol': ticker_symbol,
            'search_timestamp': datetime.utcnow().isoformat(),
            'news': [],
            'total_news': 0
        }
        
        if 'results' in response:
            for article in response['results']:
                processed_article = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'description': article.get('description', ''),
                    'published': article.get('age', ''),
                    'source': article.get('profile', {}).get('name', ''),
                    'relevance_score': self._calculate_relevance_score(article, ticker_symbol)
                }
                processed['news'].append(processed_article)
            
            # Sort by relevance and recency
            processed['news'].sort(
                key=lambda x: (x['relevance_score'], x.get('published', '')), 
                reverse=True
            )
            processed['total_news'] = len(processed['news'])
        
        return processed
    
    def _process_company_results(self, response: Dict, company_name: str) -> Dict:
        """Process company information search results"""
        processed = {
            'company_name': company_name,
            'search_timestamp': datetime.utcnow().isoformat(),
            'company_info': [],
            'total_results': 0
        }
        
        if 'web' in response and 'results' in response['web']:
            for result in response['web']['results']:
                processed_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'description': result.get('description', ''),
                    'relevance_score': self._calculate_relevance_score(result, company_name)
                }
                processed['company_info'].append(processed_result)
            
            processed['total_results'] = len(processed['company_info'])
        
        return processed
    
    def _calculate_relevance_score(self, result: Dict, search_term: str) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        title = result.get('title', '').lower()
        description = result.get('description', '').lower()
        search_term_lower = search_term.lower()
        
        # Exact match in title
        if search_term_lower in title:
            score += 10.0
        
        # Exact match in description
        if search_term_lower in description:
            score += 5.0
        
        # Financial keywords bonus
        financial_keywords = [
            'financial', 'earnings', 'revenue', 'profit', 'stock', 'share',
            'market', 'trading', 'investment', 'dividend', 'analysis'
        ]
        
        for keyword in financial_keywords:
            if keyword in title:
                score += 2.0
            if keyword in description:
                score += 1.0
        
        # Hong Kong specific bonus
        hk_keywords = ['hong kong', 'hkex', 'hk stock', 'hang seng']
        for keyword in hk_keywords:
            if keyword in title or keyword in description:
                score += 3.0
        
        return score
    
    async def _store_search_history(self, query: str, search_type: str, 
                                  results_count: int, response_time_ms: int = None):
        """Store search history in database"""
        if self.database_service:
            try:
                await self.database_service.add_search_history(
                    query, search_type, results_count, response_time_ms or 0
                )
            except Exception as e:
                logger.error(f"Failed to store search history: {e}")
    
    async def get_search_statistics(self) -> Dict:
        """Get search API usage statistics"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Count recent requests
        recent_requests = [ts for ts in self._request_timestamps if ts > hour_ago]
        
        return {
            'requests_last_hour': len(recent_requests),
            'rate_limit_per_hour': self.rate_limit_per_hour,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'remaining_requests': max(0, self.rate_limit_per_hour - len(recent_requests)),
            'api_configured': bool(self.api_key),
            'session_active': self._session is not None
        }
    
    async def health_check(self) -> bool:
        """Check if search service is healthy"""
        try:
            if not self.api_key:
                return False
            
            await self.initialize()
            
            # Make a simple test request
            test_params = {'q': 'test', 'count': 1}
            response = await self._make_request(self.endpoints['web_search'], test_params)
            
            return response is not None
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
