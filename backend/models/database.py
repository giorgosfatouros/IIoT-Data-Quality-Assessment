import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
import atexit

# Global engine instance to reuse connections
_engine = None

def get_engine() -> Engine:
    """Get database engine with connection parameters from environment variables"""
    global _engine
    
    if _engine is None:
        # Get database connection parameters from environment
        db_user = os.getenv("DB_USER", "app")
        db_pass = os.getenv("DB_PASS", "app") 
        db_ip = os.getenv("DB_IP", "127.0.0.1")
        db_port = os.getenv("DB_PORT", "1529")  # Default LeanXcale port
        db_name = os.getenv("DB_NAME", "MOH")
        
        # Create connection URL using the original format that was working
        connection_url = (
            f"leanxcale://{db_user}:{db_pass}@{db_ip}:{db_port}/{db_name}"
            f"?autocommit=False&parallel=True&txn_mode=NO_CONFLICTS_NO_LOGGING"
        )
        
        # Create engine with connection pooling and proper cleanup
        _engine = create_engine(
            connection_url,
            poolclass=StaticPool,
            pool_pre_ping=True,  # Test connections before use
            pool_recycle=1800,   # Recycle connections every 30 minutes
            pool_reset_on_return='commit',  # Reset connection state
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
