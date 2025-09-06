"""
PostgreSQL Database Service for AgentInvest
Handles persistent storage of reports, sessions, and system data
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncpg
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class DatabaseService:
    """PostgreSQL database service with connection pooling and async operations"""
    
    def __init__(self):
        # Use DATABASE_URL if available (for Neon), otherwise use individual parameters
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            self.host = os.getenv('POSTGRES_HOST', 'postgres-service')
            self.port = int(os.getenv('POSTGRES_PORT', '5432'))
            self.database = os.getenv('POSTGRES_DB', 'agentinvest')
            self.user = os.getenv('POSTGRES_USER', 'agentinvest_user')
            self.password = os.getenv('POSTGRES_PASSWORD', 'agentinvest_db_pass')

        self._pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            if self.database_url:
                # Use DATABASE_URL for connection (Neon database)
                self._pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'agentinvest_backend',
                        'timezone': 'UTC'
                    }
                )
                logger.info(f"Connected to PostgreSQL using DATABASE_URL")
            else:
                # Use individual connection parameters
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    server_settings={
                        'application_name': 'agentinvest_backend',
                        'timezone': 'UTC'
                    }
                )
                logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")

            # Create tables after successful connection
            await self._create_tables()

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Database connection pool closed")

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        async with self._pool.acquire() as conn:
            # Create feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_id VARCHAR(255) NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    user_session_id VARCHAR(255),
                    source VARCHAR(100) DEFAULT 'web',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create reports table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_id VARCHAR(255) UNIQUE NOT NULL,
                    ticker_symbol VARCHAR(20) NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    metadata JSONB,
                    content TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE
                );
            """)

            # Create sessions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE
                );
            """)

            # Create indexes for better performance (with error handling)
            try:
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_report_id ON feedback(report_id);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_ticker ON reports(ticker_symbol);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON user_sessions(session_id);")
                logger.info("Database indexes created successfully")
            except Exception as e:
                logger.warning(f"Some indexes could not be created: {e}")

            logger.info("Database tables created/verified successfully")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._pool:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            yield connection
    
    # Report Management
    async def create_report(self, ticker_symbol: str, report_title: str = None) -> str:
        """Create a new report record"""
        report_id = str(uuid.uuid4())
        
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO reports (id, ticker_symbol, report_title, status, created_at)
                VALUES ($1, $2, $3, 'pending', NOW())
            """, report_id, ticker_symbol, report_title)
        
        logger.info(f"Created report {report_id} for ticker {ticker_symbol}")
        return report_id
    
    async def update_report_status(self, report_id: str, status: str, 
                                 error_message: str = None, 
                                 generation_time: int = None,
                                 file_path: str = None,
                                 file_size: int = None) -> bool:
        """Update report status and metadata"""
        try:
            async with self.get_connection() as conn:
                if status == 'completed':
                    await conn.execute("""
                        UPDATE reports 
                        SET status = $1, completed_at = NOW(), updated_at = NOW(),
                            generation_time_seconds = $2, file_path = $3, file_size_bytes = $4
                        WHERE id = $5
                    """, status, generation_time, file_path, file_size, report_id)
                elif status == 'failed':
                    await conn.execute("""
                        UPDATE reports 
                        SET status = $1, error_message = $2, updated_at = NOW()
                        WHERE id = $3
                    """, status, error_message, report_id)
                else:
                    await conn.execute("""
                        UPDATE reports 
                        SET status = $1, updated_at = NOW()
                        WHERE id = $2
                    """, status, report_id)
            
            logger.info(f"Updated report {report_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update report {report_id}: {e}")
            return False
    
    async def get_report(self, report_id: str) -> Optional[Dict]:
        """Get report by ID"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM reports WHERE id = $1
                """, report_id)
                
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get report {report_id}: {e}")
            return None
    
    async def get_reports(self, ticker_symbol: str = None, 
                         status: str = None, 
                         limit: int = 50, 
                         offset: int = 0) -> List[Dict]:
        """Get reports with optional filtering"""
        try:
            async with self.get_connection() as conn:
                query = "SELECT * FROM reports WHERE 1=1"
                params = []
                param_count = 0
                
                if ticker_symbol:
                    param_count += 1
                    query += f" AND ticker_symbol = ${param_count}"
                    params.append(ticker_symbol)
                
                if status:
                    param_count += 1
                    query += f" AND status = ${param_count}"
                    params.append(status)
                
                query += f" ORDER BY created_at DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
                params.extend([limit, offset])
                
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get reports: {e}")
            return []
    
    async def delete_old_reports(self, days: int = 30) -> int:
        """Delete reports older than specified days"""
        try:
            async with self.get_connection() as conn:
                result = await conn.execute("""
                    DELETE FROM reports 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    AND status IN ('completed', 'failed')
                """, days)
                
                deleted_count = int(result.split()[-1])
                logger.info(f"Deleted {deleted_count} old reports")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete old reports: {e}")
            return 0
    
    # Chart Management
    async def add_chart(self, report_id: str, chart_name: str, 
                       chart_type: str, file_path: str, 
                       file_size: int, metadata: Dict = None) -> str:
        """Add chart metadata to database"""
        chart_id = str(uuid.uuid4())
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO charts (id, report_id, chart_name, chart_type, 
                                      file_path, file_size_bytes, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """, chart_id, report_id, chart_name, chart_type, 
                    file_path, file_size, json.dumps(metadata or {}))
            
            logger.info(f"Added chart {chart_name} for report {report_id}")
            return chart_id
            
        except Exception as e:
            logger.error(f"Failed to add chart: {e}")
            return None
    
    async def get_report_charts(self, report_id: str) -> List[Dict]:
        """Get all charts for a report"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM charts WHERE report_id = $1 ORDER BY created_at
                """, report_id)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get charts for report {report_id}: {e}")
            return []
    
    # Session Management
    async def create_session(self, session_token: str, user_id: str = None, 
                           expires_hours: int = 24) -> str:
        """Create user session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO user_sessions (id, session_token, user_id, 
                                             created_at, expires_at, is_active)
                    VALUES ($1, $2, $3, NOW(), $4, true)
                """, session_id, session_token, user_id, expires_at)
            
            logger.info(f"Created session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    async def get_session(self, session_token: str) -> Optional[Dict]:
        """Get session by token"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM user_sessions 
                    WHERE session_token = $1 AND is_active = true 
                    AND expires_at > NOW()
                """, session_token)
                
                if row:
                    # Update last accessed
                    await conn.execute("""
                        UPDATE user_sessions SET last_accessed = NOW() 
                        WHERE id = $1
                    """, row['id'])
                    
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def update_session_preferences(self, session_token: str, 
                                       preferences: Dict) -> bool:
        """Update session preferences"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    UPDATE user_sessions 
                    SET preferences = $1, last_accessed = NOW()
                    WHERE session_token = $2 AND is_active = true
                """, json.dumps(preferences), session_token)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session preferences: {e}")
            return False
    
    # API Cache Management
    async def cache_api_response(self, cache_key: str, cache_type: str, 
                               content: str, expires_hours: int = 1) -> bool:
        """Store API response in database cache"""
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        content_hash = str(hash(content))
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO api_cache (cache_key, cache_type, content, 
                                         content_hash, created_at, expires_at)
                    VALUES ($1, $2, $3, $4, NOW(), $5)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        created_at = NOW(),
                        expires_at = EXCLUDED.expires_at,
                        hit_count = api_cache.hit_count + 1
                """, cache_key, cache_type, content, content_hash, expires_at)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache API response: {e}")
            return False
    
    async def get_cached_api_response(self, cache_key: str) -> Optional[str]:
        """Get cached API response"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT content FROM api_cache 
                    WHERE cache_key = $1 AND expires_at > NOW()
                """, cache_key)
                
                if row:
                    # Update hit count and last accessed
                    await conn.execute("""
                        UPDATE api_cache 
                        SET hit_count = hit_count + 1, last_accessed = NOW()
                        WHERE cache_key = $1
                    """, cache_key)
                    
                    return row['content']
                return None
                
        except Exception as e:
            logger.error(f"Failed to get cached API response: {e}")
            return None
    
    # Search History
    async def add_search_history(self, query_text: str, search_type: str,
                               results_count: int, response_time_ms: int,
                               session_id: str = None) -> str:
        """Add search query to history"""
        search_id = str(uuid.uuid4())
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO search_history (id, query_text, search_type, 
                                              results_count, response_time_ms, 
                                              user_session_id, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """, search_id, query_text, search_type, results_count, 
                    response_time_ms, session_id)
            
            return search_id
            
        except Exception as e:
            logger.error(f"Failed to add search history: {e}")
            return None
    
    # System Configuration
    async def get_config(self, config_key: str) -> Optional[str]:
        """Get system configuration value"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT config_value FROM system_config 
                    WHERE config_key = $1 AND is_active = true
                """, config_key)
                
                return row['config_value'] if row else None
                
        except Exception as e:
            logger.error(f"Failed to get config {config_key}: {e}")
            return None
    
    async def set_config(self, config_key: str, config_value: str, 
                        config_type: str = 'string', 
                        description: str = None) -> bool:
        """Set system configuration value"""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO system_config (config_key, config_value, 
                                             config_type, description, 
                                             created_at, updated_at, is_active)
                    VALUES ($1, $2, $3, $4, NOW(), NOW(), true)
                    ON CONFLICT (config_key) DO UPDATE SET
                        config_value = EXCLUDED.config_value,
                        config_type = EXCLUDED.config_type,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                """, config_key, config_value, config_type, description)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set config {config_key}: {e}")
            return False

    # Feedback Management Methods
    async def create_feedback(self, report_id: str, feedback_type: str, rating: Optional[int] = None,
                            comment: Optional[str] = None, user_session_id: Optional[str] = None,
                            source: str = 'web') -> str:
        """Create new feedback entry."""
        try:
            async with self._pool.acquire() as conn:
                feedback_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO feedback (id, report_id, feedback_type, rating, comment, user_session_id, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, feedback_id, report_id, feedback_type, rating, comment, user_session_id, source)

                logger.info(f"Created feedback {feedback_id} for report {report_id}")
                return feedback_id
        except Exception as e:
            logger.error(f"Failed to create feedback: {e}")
            raise

    async def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        """Get feedback by ID."""
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM feedback WHERE id = $1", feedback_id)
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get feedback {feedback_id}: {e}")
            return None

    async def get_all_feedback(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Get all feedback with pagination."""
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM feedback
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """, limit, offset)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get all feedback: {e}")
            return []

    async def get_feedback_by_report(self, report_id: str) -> List[Dict]:
        """Get all feedback for a specific report."""
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM feedback
                    WHERE report_id = $1
                    ORDER BY created_at DESC
                """, report_id)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get feedback for report {report_id}: {e}")
            return []
    
    # Health Check
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except:
            return False
    
    # Statistics
    async def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            async with self.get_connection() as conn:
                stats = {}
                
                # Report statistics
                stats['reports'] = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'processing') as processing,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed
                    FROM reports
                """)
                
                # Recent activity
                stats['recent_reports'] = await conn.fetchval("""
                    SELECT COUNT(*) FROM reports 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                
                # Active sessions
                stats['active_sessions'] = await conn.fetchval("""
                    SELECT COUNT(*) FROM user_sessions 
                    WHERE is_active = true AND expires_at > NOW()
                """)
                
                # Cache statistics
                stats['cache_entries'] = await conn.fetchval("""
                    SELECT COUNT(*) FROM api_cache WHERE expires_at > NOW()
                """)
                
                return {k: dict(v) if hasattr(v, 'keys') else v for k, v in stats.items()}
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
