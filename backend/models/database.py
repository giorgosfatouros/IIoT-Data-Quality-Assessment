import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def get_engine() -> Engine:
    """Get database engine with connection parameters from environment variables"""
    
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
    
    # Create and return engine
    engine = create_engine(connection_url)
    return engine
