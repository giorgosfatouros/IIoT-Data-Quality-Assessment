import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
import atexit

# Global engine instance to reuse connections
_engine = None

def get_engine() -> Engine:
    """Get database engine with connection parameters from environment variables
    
    Supports PostgreSQL/TimescaleDB connection. Uses DB_HOST (preferred) or DB_IP (legacy)
    for backward compatibility during migration.
    """
    global _engine
    
    if _engine is None:
        # Get database connection parameters from environment
        # Support both DB_HOST (new) and DB_IP (legacy) for backward compatibility
        db_host = os.getenv("DB_HOST") or os.getenv("DB_IP", "127.0.0.1")
        db_user = os.getenv("DB_USER", "iiot_user")
        db_pass = os.getenv("DB_PASS", "iiot_password")
        db_port = os.getenv("DB_PORT", "5432")  # Default PostgreSQL port
        db_name = os.getenv("DB_NAME", "iiot_dqa")
        
        # Create PostgreSQL connection URL
        connection_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        
        # Create engine with connection pooling optimized for PostgreSQL
        _engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=10,  # Number of connections to maintain
            max_overflow=20,  # Maximum overflow connections
            pool_pre_ping=True,  # Test connections before use
            pool_recycle=3600,   # Recycle connections every hour
            echo=False           # Set to True for SQL debugging
        )
        
        # Register cleanup function
        atexit.register(cleanup_engine)
    
    return _engine

def cleanup_engine():
    """Properly close database connections on application shutdown"""
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception as e:
            # Ignore cleanup errors to prevent the OSError warnings
            pass
        finally:
            _engine = None
