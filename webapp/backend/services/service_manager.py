"""
Service Manager for AgentInvest Backend
Manages initialization and lifecycle of all services
"""

import logging
import asyncio
from typing import Optional
from .cache_service import CacheService
from .database_service import DatabaseService
from .message_broker import MessageBroker
from .search_service import SearchService

logger = logging.getLogger(__name__)

class ServiceManager:
    """Centralized service manager for all backend services"""
    
    def __init__(self):
        self.cache_service: Optional[CacheService] = None
        self.database_service: Optional[DatabaseService] = None
        self.message_broker: Optional[MessageBroker] = None
        self.search_service: Optional[SearchService] = None
        
        self._initialized = False
        self._shutdown = False
    
    async def initialize(self):
        """Initialize all services"""
        if self._initialized:
            logger.warning("Services already initialized")
            return
        
        try:
            logger.info("Initializing AgentInvest services...")
            
            # Initialize cache service
            logger.info("Initializing cache service...")
            self.cache_service = CacheService()
            if not self.cache_service.health_check():
                logger.warning("Cache service health check failed, but continuing...")
            
            # Initialize database service
            logger.info("Initializing database service...")
            self.database_service = DatabaseService()
            await self.database_service.initialize()
            
            # Initialize message broker
            logger.info("Initializing message broker...")
            self.message_broker = MessageBroker()
            await self.message_broker.connect()
            
            # Initialize search service with dependencies
            logger.info("Initializing search service...")
            self.search_service = SearchService(
                cache_service=self.cache_service,
                database_service=self.database_service
            )
            await self.search_service.initialize()
            
            self._initialized = True
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown all services gracefully"""
        if self._shutdown:
            return
        
        logger.info("Shutting down AgentInvest services...")
        self._shutdown = True
        
        # Shutdown in reverse order
        if self.search_service:
            try:
                await self.search_service.close()
                logger.info("Search service shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down search service: {e}")
        
        if self.message_broker:
            try:
                await self.message_broker.disconnect()
                logger.info("Message broker shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down message broker: {e}")
        
        if self.database_service:
            try:
                await self.database_service.close()
                logger.info("Database service shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down database service: {e}")
        
        # Cache service doesn't need async shutdown
        logger.info("All services shutdown complete")
    
    async def health_check(self) -> dict:
        """Check health of all services"""
        health_status = {
            'overall': True,
            'services': {}
        }
        
        # Check cache service
        if self.cache_service:
            cache_healthy = self.cache_service.health_check()
            health_status['services']['cache'] = {
                'healthy': cache_healthy,
                'stats': self.cache_service.get_cache_stats() if cache_healthy else None
            }
            if not cache_healthy:
                health_status['overall'] = False
        else:
            health_status['services']['cache'] = {'healthy': False, 'error': 'Not initialized'}
            health_status['overall'] = False
        
        # Check database service
        if self.database_service:
            try:
                db_healthy = await self.database_service.health_check()
                health_status['services']['database'] = {
                    'healthy': db_healthy,
                    'stats': await self.database_service.get_statistics() if db_healthy else None
                }
                if not db_healthy:
                    health_status['overall'] = False
            except Exception as e:
                health_status['services']['database'] = {'healthy': False, 'error': str(e)}
                health_status['overall'] = False
        else:
            health_status['services']['database'] = {'healthy': False, 'error': 'Not initialized'}
            health_status['overall'] = False
        
        # Check message broker
        if self.message_broker:
            try:
                mq_healthy = await self.message_broker.health_check()
                health_status['services']['message_broker'] = {
                    'healthy': mq_healthy,
                    'stats': await self.message_broker.get_queue_stats() if mq_healthy else None
                }
                if not mq_healthy:
                    health_status['overall'] = False
            except Exception as e:
                health_status['services']['message_broker'] = {'healthy': False, 'error': str(e)}
                health_status['overall'] = False
        else:
            health_status['services']['message_broker'] = {'healthy': False, 'error': 'Not initialized'}
            health_status['overall'] = False
        
        # Check search service
        if self.search_service:
            try:
                search_healthy = await self.search_service.health_check()
                health_status['services']['search'] = {
                    'healthy': search_healthy,
                    'stats': await self.search_service.get_search_statistics() if search_healthy else None
                }
                if not search_healthy:
                    health_status['overall'] = False
            except Exception as e:
                health_status['services']['search'] = {'healthy': False, 'error': str(e)}
                health_status['overall'] = False
        else:
            health_status['services']['search'] = {'healthy': False, 'error': 'Not initialized'}
            health_status['overall'] = False
        
        return health_status
    
    def get_cache_service(self) -> Optional[CacheService]:
        """Get cache service instance"""
        return self.cache_service
    
    def get_database_service(self) -> Optional[DatabaseService]:
        """Get database service instance"""
        return self.database_service
    
    def get_message_broker(self) -> Optional[MessageBroker]:
        """Get message broker instance"""
        return self.message_broker
    
    def get_search_service(self) -> Optional[SearchService]:
        """Get search service instance"""
        return self.search_service
    
    @property
    def is_initialized(self) -> bool:
        """Check if services are initialized"""
        return self._initialized
    
    @property
    def is_shutdown(self) -> bool:
        """Check if services are shutdown"""
        return self._shutdown

# Global service manager instance
service_manager = ServiceManager()
