"""
Redis Cache Service for AgentInvest
Handles caching of reports, charts, API responses, and search results
"""

import json
import logging
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis
import pickle
import os

logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service with TTL management and cache invalidation"""
    
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'redis-service')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', 'agentinvest_redis_pass')
        self.redis_db = int(os.getenv('REDIS_DB', '0'))
        
        # TTL configurations (in seconds)
        self.ttl_config = {
            'reports': int(os.getenv('CACHE_TTL_REPORTS', '86400')),      # 24 hours
            'charts': int(os.getenv('CACHE_TTL_CHARTS', '43200')),        # 12 hours
            'api_responses': int(os.getenv('CACHE_TTL_API', '3600')),     # 1 hour
            'search_results': int(os.getenv('CACHE_TTL_SEARCH', '1800')), # 30 minutes
            'sessions': int(os.getenv('CACHE_TTL_SESSIONS', '7200')),     # 2 hours
        }
        
        self._redis_client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection with retry logic"""
        try:
            self._redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                db=self.redis_db,
                decode_responses=False,  # We'll handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self._redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis_client = None
    
    def _ensure_connection(self):
        """Ensure Redis connection is active"""
        if not self._redis_client:
            self._connect()
        
        try:
            self._redis_client.ping()
        except:
            logger.warning("Redis connection lost, reconnecting...")
            self._connect()
    
    def _generate_key(self, cache_type: str, identifier: str, **kwargs) -> str:
        """Generate cache key with namespace and optional parameters"""
        namespace = f"agentinvest:{cache_type}"
        
        if kwargs:
            # Include additional parameters in key
            params = "&".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
            return f"{namespace}:{identifier}:{params}"
        
        return f"{namespace}:{identifier}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        if isinstance(data, (str, bytes)):
            return data.encode() if isinstance(data, str) else data
        
        # Use pickle for complex objects
        return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        if not data:
            return None
        
        try:
            # Try to decode as string first
            return data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Try pickle for complex objects
                return pickle.loads(data)
            except:
                # Return raw bytes if all else fails
                return data
    
    # Report Caching
    def cache_report(self, report_id: str, content: str, content_type: str = 'html') -> bool:
        """Cache a generated report"""
        try:
            self._ensure_connection()
            key = self._generate_key('reports', report_id, type=content_type)
            
            # Store report content
            self._redis_client.setex(
                key,
                self.ttl_config['reports'],
                self._serialize_data(content)
            )
            
            # Store metadata
            metadata_key = self._generate_key('reports:meta', report_id)
            metadata = {
                'cached_at': datetime.utcnow().isoformat(),
                'content_type': content_type,
                'size': len(content),
                'ttl': self.ttl_config['reports']
            }
            
            self._redis_client.setex(
                metadata_key,
                self.ttl_config['reports'],
                json.dumps(metadata)
            )
            
            logger.info(f"Cached report {report_id} ({content_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache report {report_id}: {e}")
            return False
    
    def get_cached_report(self, report_id: str, content_type: str = 'html') -> Optional[str]:
        """Retrieve cached report"""
        try:
            self._ensure_connection()
            key = self._generate_key('reports', report_id, type=content_type)

            data = self._redis_client.get(key)
            if data:
                logger.info(f"Cache hit for report {report_id}")
                return self._deserialize_data(data)

            logger.info(f"Cache miss for report {report_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get cached report {report_id}: {e}")
            return None

    def get_cached_report_with_miss_callback(self, report_id: str, content_type: str = 'html',
                                           miss_callback=None, ticker: str = None) -> Optional[str]:
        """Retrieve cached report with cache miss callback for async generation"""
        try:
            self._ensure_connection()
            key = self._generate_key('reports', report_id, type=content_type)

            data = self._redis_client.get(key)
            if data:
                logger.info(f"Cache hit for report {report_id}")
                return self._deserialize_data(data)
            else:
                logger.info(f"Cache miss for report {report_id}")

                # Trigger cache miss callback for async processing
                if miss_callback and ticker:
                    try:
                        miss_callback(ticker, report_id)
                        logger.info(f"Triggered async report generation for {ticker} via cache miss")
                    except Exception as e:
                        logger.error(f"Failed to trigger async report generation: {e}")

                return None

        except Exception as e:
            logger.error(f"Failed to get cached report {report_id}: {e}")
            return None
    
    # Chart Caching
    def cache_chart(self, report_id: str, chart_name: str, chart_data: bytes) -> bool:
        """Cache chart image data"""
        try:
            self._ensure_connection()
            key = self._generate_key('charts', f"{report_id}:{chart_name}")
            
            self._redis_client.setex(
                key,
                self.ttl_config['charts'],
                chart_data
            )
            
            logger.info(f"Cached chart {chart_name} for report {report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache chart {chart_name}: {e}")
            return False
    
    def get_cached_chart(self, report_id: str, chart_name: str) -> Optional[bytes]:
        """Retrieve cached chart"""
        try:
            self._ensure_connection()
            key = self._generate_key('charts', f"{report_id}:{chart_name}")
            
            data = self._redis_client.get(key)
            if data:
                logger.info(f"Cache hit for chart {chart_name}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached chart {chart_name}: {e}")
            return None
    
    # API Response Caching
    def cache_api_response(self, endpoint: str, params: Dict, response_data: Any) -> bool:
        """Cache API response with parameter-based key"""
        try:
            self._ensure_connection()
            
            # Create hash of parameters for consistent key generation
            param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
            key = self._generate_key('api', f"{endpoint}:{param_hash}")
            
            cache_data = {
                'data': response_data,
                'cached_at': datetime.utcnow().isoformat(),
                'params': params
            }
            
            self._redis_client.setex(
                key,
                self.ttl_config['api_responses'],
                self._serialize_data(cache_data)
            )
            
            logger.info(f"Cached API response for {endpoint}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache API response for {endpoint}: {e}")
            return False
    
    def get_cached_api_response(self, endpoint: str, params: Dict) -> Optional[Any]:
        """Retrieve cached API response"""
        try:
            self._ensure_connection()
            
            param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
            key = self._generate_key('api', f"{endpoint}:{param_hash}")
            
            data = self._redis_client.get(key)
            if data:
                cache_data = self._deserialize_data(data)
                if isinstance(cache_data, dict) and 'data' in cache_data:
                    logger.info(f"Cache hit for API {endpoint}")
                    return cache_data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached API response for {endpoint}: {e}")
            return None
    
    # Search Results Caching
    def cache_search_results(self, query: str, results: List[Dict]) -> bool:
        """Cache search results"""
        try:
            self._ensure_connection()
            
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            key = self._generate_key('search', query_hash)
            
            cache_data = {
                'query': query,
                'results': results,
                'cached_at': datetime.utcnow().isoformat(),
                'count': len(results)
            }
            
            self._redis_client.setex(
                key,
                self.ttl_config['search_results'],
                self._serialize_data(cache_data)
            )
            
            logger.info(f"Cached search results for query: {query}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
            return False
    
    def get_cached_search_results(self, query: str) -> Optional[List[Dict]]:
        """Retrieve cached search results"""
        try:
            self._ensure_connection()
            
            query_hash = hashlib.md5(query.lower().encode()).hexdigest()
            key = self._generate_key('search', query_hash)
            
            data = self._redis_client.get(key)
            if data:
                cache_data = self._deserialize_data(data)
                if isinstance(cache_data, dict) and 'results' in cache_data:
                    logger.info(f"Cache hit for search: {query}")
                    return cache_data['results']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached search results: {e}")
            return None
    
    # Cache Invalidation
    def invalidate_report_cache(self, report_id: str) -> bool:
        """Invalidate all cached data for a specific report"""
        try:
            self._ensure_connection()
            
            # Find all keys related to this report
            patterns = [
                self._generate_key('reports', f"{report_id}*"),
                self._generate_key('reports:meta', f"{report_id}*"),
                self._generate_key('charts', f"{report_id}*")
            ]
            
            deleted_count = 0
            for pattern in patterns:
                keys = self._redis_client.keys(pattern)
                if keys:
                    deleted_count += self._redis_client.delete(*keys)
            
            logger.info(f"Invalidated {deleted_count} cache entries for report {report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache for report {report_id}: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        try:
            self._ensure_connection()
            
            keys = self._redis_client.keys(f"agentinvest:{pattern}")
            if keys:
                deleted_count = self._redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache pattern {pattern}: {e}")
            return 0
    
    # Health and Statistics
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            self._ensure_connection()
            
            info = self._redis_client.info()
            
            # Count keys by type
            key_counts = {}
            for cache_type in ['reports', 'charts', 'api', 'search']:
                pattern = f"agentinvest:{cache_type}:*"
                keys = self._redis_client.keys(pattern)
                key_counts[cache_type] = len(keys)
            
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            hit_rate = hits / max(hits + misses, 1)

            return {
                'connected': True,
                'memory_used': info.get('used_memory_human', 'Unknown'),
                'total_keys': info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
                'key_counts': key_counts,
                'hit_rate': hit_rate,
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def health_check(self) -> bool:
        """Check if cache service is healthy"""
        try:
            self._ensure_connection()
            self._redis_client.ping()
            return True
        except:
            return False

    def invalidate_pattern(self, pattern: str) -> bool:
        """Invalidate cache entries matching a pattern"""
        try:
            self._ensure_connection()

            # Get all keys matching the pattern
            keys = []
            for key in self._redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                # Delete all matching keys
                deleted_count = self._redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache entries matching pattern: {pattern}")
                return True
            else:
                logger.info(f"No cache entries found matching pattern: {pattern}")
                return True

        except Exception as e:
            logger.error(f"Failed to invalidate cache pattern {pattern}: {e}")
            return False
