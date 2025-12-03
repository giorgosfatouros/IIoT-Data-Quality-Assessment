import time
import logging
from typing import Optional, Callable, Any
from functools import wraps
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError
from psycopg2 import OperationalError as Psycopg2OperationalError
from models.database import get_engine

logger = logging.getLogger(__name__)

class ConnectionRetryManager:
    """Manages database connection retries with exponential backoff and circuit breaker pattern"""
    
    def __init__(self, max_retries: int = 10, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (too many failures in short time)"""
        if self.failure_count >= self.circuit_breaker_threshold:
            if time.time() - self.last_failure_time < self.circuit_breaker_timeout:
                return True
            else:
                # Reset circuit breaker after timeout
                self.failure_count = 0
        return False
    
    def record_failure(self):
        """Record a connection failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(f"Connection failure recorded. Total failures: {self.failure_count}")
    
    def record_success(self):
        """Record a successful connection"""
        if self.failure_count > 0:
            logger.info(f"Connection restored after {self.failure_count} failures")
        self.failure_count = 0
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        return max(0.1, delay + jitter)
    
    def test_connection(self) -> bool:
        """Test if database connection is healthy"""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.debug(f"Connection test failed: {e}")
            return False

# Global retry manager instance
retry_manager = ConnectionRetryManager()

def with_retry(max_retries: Optional[int] = None, test_connection: bool = True):
    """
    Decorator for database operations with automatic retry logic
    
    Args:
        max_retries: Override default max retries
        test_connection: Whether to test connection health before retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries if max_retries is not None else retry_manager.max_retries
            
            for attempt in range(retries + 1):
                try:
                    # Check circuit breaker
                    if retry_manager.is_circuit_open():
                        raise Exception("Circuit breaker is open - too many recent failures")
                    
                    # Test connection health if requested
                    if test_connection and attempt > 0:
                        if not retry_manager.test_connection():
                            logger.warning("Connection health check failed, will retry")
                            if attempt < retries:
                                delay = retry_manager.get_retry_delay(attempt)
                                logger.info(f"Waiting {delay:.2f}s before retry {attempt + 1}/{retries}")
                                time.sleep(delay)
                                continue
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    retry_manager.record_success()
                    return result
                    
                except (SQLAlchemyError, DisconnectionError, OperationalError, Psycopg2OperationalError, Exception) as e:
                    retry_manager.record_failure()
                    
                    # Check if it's a connection error that should be retried
                    is_retryable = isinstance(e, (DisconnectionError, OperationalError, Psycopg2OperationalError)) or \
                                  "connection" in str(e).lower() or \
                                  "network" in str(e).lower()
                    
                    if attempt < retries and is_retryable:
                        delay = retry_manager.get_retry_delay(attempt)
                        logger.warning(f"Database operation failed (attempt {attempt + 1}/{retries + 1}): {e}")
                        logger.info(f"Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Database operation failed after {retries + 1} attempts: {e}")
                        raise
            
        return wrapper
    return decorator

def get_connection_with_retry():
    """Get a database connection with retry logic"""
    @with_retry()
    def _get_connection():
        engine = get_engine()
        return engine.connect()
    
    return _get_connection()

def execute_with_retry(query, params=None):
    """Execute a query with retry logic"""
    @with_retry()
    def _execute():
        engine = get_engine()
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result.fetchall()
    
    return _execute()
