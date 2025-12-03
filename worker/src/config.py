"""
Configuration management for Data Quality Worker
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class WorkerConfig(BaseSettings):
    """Worker configuration loaded from environment variables"""

    # Database Configuration
    db_host: str = Field(default="timescaledb", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="iiot_dqa", alias="DB_NAME")
    db_user: str = Field(default="iiot_user", alias="DB_USER")
    db_pass: str = Field(default="iiot_password", alias="DB_PASS")

    # Data Quality Configuration
    aggregation_interval_seconds: int = Field(default=3600, alias="AGGREGATION_INTERVAL_SECONDS")
    original_frequency_seconds: int = Field(default=10, alias="ORIGINAL_FREQUENCY_SECONDS")
    anomaly_threshold_zscore: float = Field(default=2.0, alias="ANOMALY_THRESHOLD_ZSCORE")

    # Worker Configuration
    batch_size: int = Field(default=1000, alias="BATCH_SIZE")
    worker_interval_seconds: int = Field(default=60, alias="WORKER_INTERVAL_SECONDS")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        populate_by_name = True

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def expected_readings_per_interval(self) -> int:
        """Calculate expected number of readings per aggregation interval"""
        return self.aggregation_interval_seconds // self.original_frequency_seconds


# Global config instance
config = WorkerConfig()




