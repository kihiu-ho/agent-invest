"""
Historical Data Cache Manager

Implements robust caching system for Yahoo Finance historical data with:
- File-based caching with naming convention {ticker}_{download_timestamp}.json
- Smart data retrieval logic with cache-first approach
- Cache expiration and refresh logic
- Metadata tracking and monitoring
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)

class HistoricalDataCache:
    """
    Manages file-based caching for Yahoo Finance historical data with intelligent retrieval.
    """
    
    def __init__(self, cache_dir: str = "historical_data_cache", cache_expiry_hours: int = 24):
        """
        Initialize the historical data cache manager.
        
        Args:
            cache_dir: Directory to store cached data files
            cache_expiry_hours: Hours after which cached data is considered stale
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata tracking
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"✅ Historical data cache initialized: {self.cache_dir}")
        logger.info(f"📅 Cache expiry: {cache_expiry_hours} hours")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "cache_stats": {
                "total_files": 0,
                "total_size_mb": 0,
                "last_cleanup": None
            },
            "ticker_index": {}
        }
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_filename(self, ticker: str, timestamp: Optional[datetime] = None) -> str:
        """
        Generate cache filename with format: {ticker}_{download_timestamp}.json
        
        Args:
            ticker: Stock ticker symbol
            timestamp: Download timestamp (defaults to current time)
            
        Returns:
            Cache filename string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format timestamp as YYYYMMDD_HHMMSS
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Clean ticker for filename (replace special characters)
        clean_ticker = ticker.replace(".", "_").replace(":", "_")
        
        return f"{clean_ticker}_{timestamp_str}.json"
    
    def _find_latest_cache_file(self, ticker: str) -> Optional[Tuple[Path, datetime]]:
        """
        Find the most recent cache file for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (file_path, creation_time) or None if not found
        """
        clean_ticker = ticker.replace(".", "_").replace(":", "_")
        pattern = f"{clean_ticker}_*.json"
        
        cache_files = []
        for file_path in self.cache_dir.glob(pattern):
            try:
                # Extract timestamp from filename
                filename = file_path.stem
                timestamp_part = filename.split('_', 1)[1]  # Get part after first underscore
                file_time = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                cache_files.append((file_path, file_time))
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping invalid cache file {file_path}: {e}")
                continue
        
        if cache_files:
            # Return the most recent file
            return max(cache_files, key=lambda x: x[1])
        
        return None
    
    def _is_cache_valid(self, file_time: datetime) -> bool:
        """
        Check if cached data is still valid based on expiry time.
        
        Args:
            file_time: Creation time of the cache file
            
        Returns:
            True if cache is still valid, False if expired
        """
        expiry_time = file_time + timedelta(hours=self.cache_expiry_hours)
        return datetime.now() < expiry_time
    
    async def get_cached_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached historical data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Cached data dictionary or None if not found/expired
        """
        try:
            cache_result = self._find_latest_cache_file(ticker)
            
            if cache_result is None:
                self.cache_misses += 1
                logger.info(f"📊 Cache miss for {ticker}: No cache file found")
                return None
            
            file_path, file_time = cache_result
            
            if not self._is_cache_valid(file_time):
                self.cache_misses += 1
                logger.info(f"📊 Cache miss for {ticker}: Cache expired ({file_time})")
                return None
            
            # Load cached data
            with open(file_path, 'r') as f:
                cached_data = json.load(f)
            
            self.cache_hits += 1
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            logger.info(f"✅ Cache hit for {ticker}: {file_path.name} (age: {age_hours:.1f}h)")
            
            # Add cache metadata to response
            cached_data['_cache_info'] = {
                'cached_at': file_time.isoformat(),
                'cache_file': file_path.name,
                'age_hours': round(age_hours, 1),
                'from_cache': True
            }
            
            return cached_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving cached data for {ticker}: {e}")
            return None
    
    async def store_data(self, ticker: str, data: Dict[str, Any], 
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store historical data in cache with metadata.
        
        Args:
            ticker: Stock ticker symbol
            data: Historical data to cache
            metadata: Optional metadata about the data
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            timestamp = datetime.now()
            filename = self._generate_cache_filename(ticker, timestamp)
            file_path = self.cache_dir / filename
            
            # Prepare data for storage
            cache_data = {
                "ticker": ticker,
                "download_timestamp": timestamp.isoformat(),
                "data_source": "yahoo_finance",
                "metadata": metadata or {},
                "data": data
            }
            
            # Add data quality metrics
            if isinstance(data, dict) and 'historical_data' in data:
                hist_data = data['historical_data']
                cache_data["metadata"].update({
                    "data_points": hist_data.get('data_points', 0),
                    "date_range": {
                        "start": hist_data.get('start_date'),
                        "end": hist_data.get('end_date')
                    },
                    "period": hist_data.get('period', 'unknown')
                })
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            # Update metadata index
            self._update_metadata_index(ticker, filename, timestamp, cache_data)
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Cached data for {ticker}: {filename} ({file_size_mb:.2f}MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing cache data for {ticker}: {e}")
            return False
    
    def _update_metadata_index(self, ticker: str, filename: str, 
                              timestamp: datetime, cache_data: Dict[str, Any]):
        """Update the metadata index with new cache entry."""
        try:
            self.metadata["ticker_index"][ticker] = {
                "latest_file": filename,
                "last_updated": timestamp.isoformat(),
                "data_points": cache_data.get("metadata", {}).get("data_points", 0),
                "period": cache_data.get("metadata", {}).get("period", "unknown")
            }
            
            # Update cache stats
            self.metadata["cache_stats"]["total_files"] = len(list(self.cache_dir.glob("*.json")))
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
            self.metadata["cache_stats"]["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to update metadata index: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and performance metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "api_calls": self.api_calls,
                "hit_rate_percent": round(hit_rate, 1),
                "total_requests": total_requests
            },
            "cache_storage": self.metadata.get("cache_stats", {}),
            "cached_tickers": len(self.metadata.get("ticker_index", {}))
        }
    
    async def cleanup_expired_cache(self) -> Dict[str, int]:
        """Remove expired cache files and return cleanup statistics."""
        try:
            removed_files = 0
            freed_space_mb = 0
            
            for file_path in self.cache_dir.glob("*.json"):
                if file_path.name == "cache_metadata.json":
                    continue
                
                try:
                    # Extract timestamp from filename
                    filename = file_path.stem
                    if '_' not in filename:
                        continue
                    
                    timestamp_part = filename.split('_', 1)[1]
                    file_time = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                    
                    if not self._is_cache_valid(file_time):
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        removed_files += 1
                        freed_space_mb += file_size / (1024 * 1024)
                        
                except Exception as e:
                    logger.debug(f"Error processing cache file {file_path}: {e}")
                    continue
            
            # Update metadata
            self.metadata["cache_stats"]["last_cleanup"] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"🧹 Cache cleanup: removed {removed_files} files, freed {freed_space_mb:.2f}MB")
            
            return {
                "removed_files": removed_files,
                "freed_space_mb": round(freed_space_mb, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
            return {"removed_files": 0, "freed_space_mb": 0}
