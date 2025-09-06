#!/usr/bin/env python3
"""
Quick script to query and display database content for verification.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path.cwd().parent))

from database_manager import db_manager

async def query_database_content():
    """Query and display database content."""
    print("💾 Querying Database Content")
    print("="*50)
    
    if not db_manager.available:
        print("❌ Database not available")
        return
    
    await db_manager.initialize()
    
    try:
        # Query all cached content
        query = """
        SELECT ticker, source, page_type, content_length, extraction_method, 
               scraped_date, SUBSTRING(markdown_content, 1, 200) as content_preview
        FROM web_scraping_cache 
        ORDER BY scraped_date DESC
        """
        
        async with db_manager.connection_pool.acquire() as connection:
            rows = await connection.fetch(query)
        
        if rows:
            print(f"Found {len(rows)} cached entries:\n")
            
            for i, row in enumerate(rows, 1):
                print(f"Entry {i}:")
                print(f"  Ticker: {row['ticker']}")
                print(f"  Source: {row['source']}")
                print(f"  Page Type: {row['page_type']}")
                print(f"  Content Length: {row['content_length']:,} chars")
                print(f"  Extraction Method: {row['extraction_method']}")
                print(f"  Scraped Date: {row['scraped_date']}")
                print(f"  Content Preview: {row['content_preview'][:100]}...")
                print()
        else:
            print("No cached entries found")
    
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(query_database_content())
