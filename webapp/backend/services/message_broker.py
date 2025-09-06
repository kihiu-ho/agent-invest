"""
RabbitMQ Message Broker Service for AgentInvest
Handles async communication between microservices
"""

import json
import logging
import os
import asyncio
from typing import Dict, Any, Callable, Optional
import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType
from datetime import datetime

logger = logging.getLogger(__name__)

class MessageBroker:
    """RabbitMQ message broker for microservice communication"""

    # Class-level connection tracking to prevent duplicate connections
    _shared_connection = None
    _shared_channel = None
    _shared_exchanges = {}
    _shared_queues = {}
    _connection_count = 0
    _bindings_initialized = False

    def __init__(self):
        self.host = os.getenv('RABBITMQ_HOST', 'rabbitmq-service')
        self.port = int(os.getenv('RABBITMQ_PORT', '5672'))
        self.username = os.getenv('RABBITMQ_USER', 'agentinvest_user')
        self.password = os.getenv('RABBITMQ_PASSWORD', 'agentinvest_mq_pass')
        self.vhost = os.getenv('RABBITMQ_VHOST', 'agentinvest')

        # Use shared connection if available
        self._connection = MessageBroker._shared_connection
        self._channel = MessageBroker._shared_channel
        self._exchanges = MessageBroker._shared_exchanges
        self._queues = MessageBroker._shared_queues
        self._consumers = {}
        
        # Queue configurations
        self.queue_configs = {
            'report_generation': {
                'durable': True,
                'arguments': {
                    'x-message-ttl': 3600000,  # 1 hour
                    'x-max-length': 1000,
                    'x-dead-letter-exchange': 'dlx',
                    'x-dead-letter-routing-key': 'report_generation.failed'
                }
            },
            'cache_invalidation': {
                'durable': True,
                'arguments': {
                    'x-message-ttl': 300000,  # 5 minutes
                    'x-max-length': 5000
                }
            },
            'api_requests': {
                'durable': True,
                'arguments': {
                    'x-message-ttl': 1800000,  # 30 minutes
                    'x-max-length': 2000,
                    'x-dead-letter-exchange': 'dlx',
                    'x-dead-letter-routing-key': 'api_requests.failed'
                }
            },
            'search_requests': {
                'durable': True,
                'arguments': {
                    'x-message-ttl': 600000,  # 10 minutes
                    'x-max-length': 3000
                }
            },
            'performance_metrics': {
                'durable': True,
                'arguments': {
                    'x-message-ttl': 86400000,  # 24 hours
                    'x-max-length': 10000
                }
            }
        }
    
    async def connect(self):
        """Establish connection to RabbitMQ with shared connection management"""
        try:
            # Check if shared connection already exists and is valid
            if (MessageBroker._shared_connection and
                not MessageBroker._shared_connection.is_closed):

                # Reuse existing connection
                self._connection = MessageBroker._shared_connection
                self._channel = MessageBroker._shared_channel
                self._exchanges = MessageBroker._shared_exchanges
                self._queues = MessageBroker._shared_queues

                MessageBroker._connection_count += 1
                logger.info(f"Reusing existing RabbitMQ connection (count: {MessageBroker._connection_count})")
                return

            # Create new shared connection
            connection_url = f"amqp://{self.username}:{self.password}@{self.host}:{self.port}/{self.vhost}"

            MessageBroker._shared_connection = await aio_pika.connect_robust(
                connection_url,
                heartbeat=60,
                blocked_connection_timeout=300,
                connection_attempts=5,
                retry_delay=5
            )

            MessageBroker._shared_channel = await MessageBroker._shared_connection.channel()
            await MessageBroker._shared_channel.set_qos(prefetch_count=10)

            # Update instance references
            self._connection = MessageBroker._shared_connection
            self._channel = MessageBroker._shared_channel
            self._exchanges = MessageBroker._shared_exchanges
            self._queues = MessageBroker._shared_queues

            # Setup exchanges and queues only once
            if not MessageBroker._bindings_initialized:
                await self._setup_exchanges()
                await self._setup_queues()
                MessageBroker._bindings_initialized = True
                logger.info("✅ RabbitMQ exchanges, queues, and bindings initialized")

            MessageBroker._connection_count = 1
            logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Close RabbitMQ connection with shared connection management"""
        try:
            MessageBroker._connection_count -= 1
            logger.info(f"Disconnecting from RabbitMQ (remaining connections: {MessageBroker._connection_count})")

            # Only close the shared connection when no more instances are using it
            if MessageBroker._connection_count <= 0:
                if (MessageBroker._shared_connection and
                    not MessageBroker._shared_connection.is_closed):
                    await MessageBroker._shared_connection.close()
                    logger.info("✅ Closed shared RabbitMQ connection")

                # Reset shared state
                MessageBroker._shared_connection = None
                MessageBroker._shared_channel = None
                MessageBroker._shared_exchanges = {}
                MessageBroker._shared_queues = {}
                MessageBroker._bindings_initialized = False
                MessageBroker._connection_count = 0

        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    async def _setup_exchanges(self):
        """Setup RabbitMQ exchanges"""
        exchanges = [
            ('agentinvest.direct', ExchangeType.DIRECT),
            ('agentinvest.topic', ExchangeType.TOPIC),
            ('agentinvest.fanout', ExchangeType.FANOUT),
            ('dlx', ExchangeType.DIRECT)  # Dead letter exchange
        ]
        
        for exchange_name, exchange_type in exchanges:
            exchange = await self._channel.declare_exchange(
                exchange_name,
                exchange_type,
                durable=True
            )
            self._exchanges[exchange_name] = exchange
            logger.info(f"Declared exchange: {exchange_name}")
    
    async def _setup_queues(self):
        """Setup RabbitMQ queues"""
        # Setup main queues
        for queue_name, config in self.queue_configs.items():
            queue = await self._channel.declare_queue(
                queue_name,
                durable=config['durable'],
                arguments=config.get('arguments', {})
            )
            self._queues[queue_name] = queue
            logger.info(f"Declared queue: {queue_name}")
        
        # Setup dead letter queues
        dlq_names = ['dlq.report_generation', 'dlq.api_requests']
        for dlq_name in dlq_names:
            dlq = await self._channel.declare_queue(dlq_name, durable=True)
            self._queues[dlq_name] = dlq
            logger.info(f"Declared dead letter queue: {dlq_name}")
        
        # Setup bindings
        await self._setup_bindings()
    
    async def _setup_bindings(self):
        """Setup queue bindings to exchanges (idempotent)"""
        bindings = [
            ('agentinvest.direct', 'report_generation', 'report.generate'),
            ('agentinvest.direct', 'cache_invalidation', 'cache.invalidate'),
            ('agentinvest.direct', 'api_requests', 'api.request'),
            ('agentinvest.direct', 'search_requests', 'search.request'),
            ('agentinvest.topic', 'performance_metrics', 'metrics.*'),
            ('agentinvest.fanout', 'cache_invalidation', ''),
            ('dlx', 'dlq.report_generation', 'report_generation.failed'),
            ('dlx', 'dlq.api_requests', 'api_requests.failed')
        ]

        logger.info("🔗 Setting up queue bindings...")

        for exchange_name, queue_name, routing_key in bindings:
            if exchange_name in self._exchanges and queue_name in self._queues:
                try:
                    await self._queues[queue_name].bind(
                        self._exchanges[exchange_name],
                        routing_key
                    )

                    # Special logging for dead letter queues
                    if queue_name.startswith('dlq.'):
                        logger.info(f"🔗 Bound dead letter queue {queue_name} to exchange {exchange_name} with key {routing_key}")
                    else:
                        logger.info(f"🔗 Bound queue {queue_name} to exchange {exchange_name} with key {routing_key}")

                except Exception as e:
                    # Binding might already exist, which is fine
                    logger.debug(f"Binding may already exist: {queue_name} -> {exchange_name} ({e})")

        logger.info("✅ Queue bindings setup completed")
    
    async def publish_message(self, exchange_name: str, routing_key: str, 
                            message_data: Dict[str, Any], 
                            priority: int = 0,
                            expiration: int = None) -> bool:
        """Publish message to exchange"""
        try:
            if not self._connection or self._connection.is_closed:
                await self.connect()
            
            # Add metadata
            message_data['_metadata'] = {
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'agentinvest_backend',
                'message_id': f"{datetime.utcnow().timestamp()}_{routing_key}"
            }
            
            message_body = json.dumps(message_data).encode()
            
            message = Message(
                message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
                expiration=expiration,
                timestamp=datetime.utcnow(),
                headers={
                    'source': 'agentinvest_backend',
                    'routing_key': routing_key
                }
            )
            
            if exchange_name in self._exchanges:
                await self._exchanges[exchange_name].publish(
                    message,
                    routing_key=routing_key
                )
                logger.info(f"Published message to {exchange_name} with key {routing_key}")
                return True
            else:
                logger.error(f"Exchange {exchange_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def consume_messages(self, queue_name: str, 
                             callback: Callable,
                             auto_ack: bool = False) -> bool:
        """Start consuming messages from queue"""
        try:
            if not self._connection or self._connection.is_closed:
                await self.connect()
            
            if queue_name not in self._queues:
                logger.error(f"Queue {queue_name} not found")
                return False
            
            async def message_handler(message: aio_pika.IncomingMessage):
                async with message.process(ignore_processed=True):
                    try:
                        # Parse message
                        message_data = json.loads(message.body.decode())
                        
                        # Call callback
                        await callback(message_data, message)
                        
                        if not auto_ack:
                            message.ack()
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        message.nack(requeue=False)  # Send to DLQ
            
            # Start consuming
            consumer_tag = await self._queues[queue_name].consume(
                message_handler,
                no_ack=auto_ack
            )
            
            self._consumers[queue_name] = consumer_tag
            logger.info(f"Started consuming from queue: {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start consuming from {queue_name}: {e}")
            return False
    
    async def stop_consuming(self, queue_name: str) -> bool:
        """Stop consuming from queue"""
        try:
            if queue_name in self._consumers:
                await self._queues[queue_name].cancel(self._consumers[queue_name])
                del self._consumers[queue_name]
                logger.info(f"Stopped consuming from queue: {queue_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop consuming from {queue_name}: {e}")
            return False
    
    # High-level message publishing methods
    async def publish_report_generation_request(self, ticker_symbol: str, 
                                              report_id: str,
                                              priority: int = 0) -> bool:
        """Publish report generation request"""
        message_data = {
            'action': 'generate_report',
            'ticker_symbol': ticker_symbol,
            'report_id': report_id,
            'priority': priority
        }
        
        return await self.publish_message(
            'agentinvest.direct',
            'report.generate',
            message_data,
            priority=priority,
            expiration=3600000  # 1 hour
        )
    
    async def publish_cache_invalidation(self, cache_keys: list = None, 
                                       patterns: list = None,
                                       report_id: str = None) -> bool:
        """Publish cache invalidation request"""
        message_data = {
            'action': 'invalidate_cache',
            'cache_keys': cache_keys or [],
            'patterns': patterns or [],
            'report_id': report_id
        }
        
        return await self.publish_message(
            'agentinvest.fanout',  # Fanout to all cache invalidation consumers
            '',
            message_data,
            expiration=300000  # 5 minutes
        )
    
    async def publish_api_request(self, endpoint: str, params: Dict,
                                request_id: str = None) -> bool:
        """Publish API request for async processing"""
        message_data = {
            'action': 'api_request',
            'endpoint': endpoint,
            'params': params,
            'request_id': request_id or f"req_{datetime.utcnow().timestamp()}"
        }
        
        return await self.publish_message(
            'agentinvest.direct',
            'api.request',
            message_data,
            expiration=1800000  # 30 minutes
        )
    
    async def publish_search_request(self, query: str, search_type: str = 'ticker',
                                   session_id: str = None) -> bool:
        """Publish search request"""
        message_data = {
            'action': 'search_request',
            'query': query,
            'search_type': search_type,
            'session_id': session_id
        }
        
        return await self.publish_message(
            'agentinvest.direct',
            'search.request',
            message_data,
            expiration=600000  # 10 minutes
        )
    
    async def publish_performance_metric(self, metric_name: str, 
                                       metric_value: float,
                                       tags: Dict = None) -> bool:
        """Publish performance metric"""
        message_data = {
            'action': 'performance_metric',
            'metric_name': metric_name,
            'metric_value': metric_value,
            'tags': tags or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return await self.publish_message(
            'agentinvest.topic',
            f'metrics.{metric_name}',
            message_data,
            expiration=86400000  # 24 hours
        )
    
    # Health and monitoring
    async def health_check(self) -> bool:
        """Check if message broker is healthy"""
        try:
            if not self._connection or self._connection.is_closed:
                await self.connect()
            
            # Try to declare a temporary queue
            temp_queue = await self._channel.declare_queue(
                'health_check_temp',
                exclusive=True,
                auto_delete=True
            )
            await temp_queue.delete()
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        try:
            stats = {}
            
            for queue_name, queue in self._queues.items():
                # Get queue info
                queue_info = await queue.channel.queue_declare(
                    queue_name,
                    passive=True
                )
                
                stats[queue_name] = {
                    'message_count': queue_info.method.message_count,
                    'consumer_count': queue_info.method.consumer_count
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def purge_queue(self, queue_name: str) -> bool:
        """Purge all messages from queue"""
        try:
            if queue_name in self._queues:
                await self._queues[queue_name].purge()
                logger.info(f"Purged queue: {queue_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {e}")
            return False
