"""
Data Quality Worker - Main Entry Point
Runs continuous aggregation and quality assessment
"""
import logging
import sys
import time
import signal
from datetime import datetime
import os

from worker.src.config import config
from worker.src.database import db_manager
from worker.src.aggregation_service import aggregation_service
from worker.src.quality_assessment import quality_engine

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add file handler if logs directory exists
logs_dir = '/app/logs'
if os.path.exists(logs_dir):
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'worker.log'))
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


def check_database_health() -> bool:
    """
    Check if database is healthy and accessible

    Returns:
        True if database is healthy, False otherwise
    """
    try:
        return db_manager.check_health()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def wait_for_database(max_retries: int = 30, retry_interval: int = 5):
    """
    Wait for database to become available

    Args:
        max_retries: Maximum number of connection attempts
        retry_interval: Seconds to wait between retries
    """
    logger.info("Waiting for database connection...")

    for attempt in range(1, max_retries + 1):
        if check_database_health():
            logger.info("Database connection established")
            return

        logger.warning(
            f"Database not ready (attempt {attempt}/{max_retries}). "
            f"Retrying in {retry_interval} seconds..."
        )
        time.sleep(retry_interval)

    raise RuntimeError(f"Failed to connect to database after {max_retries} attempts")


def run_worker():
    """Main worker loop"""
    global shutdown_requested

    logger.info("=" * 80)
    logger.info("IIoT Data Quality Assessment Worker Starting")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Database: {config.db_host}:{config.db_port}/{config.db_name}")
    logger.info(f"  Aggregation Interval: {config.aggregation_interval_seconds}s")
    logger.info(f"  Original Frequency: {config.original_frequency_seconds}s")
    logger.info(f"  Expected Readings/Interval: {config.expected_readings_per_interval}")
    logger.info(f"  Anomaly Threshold (Z-score): {config.anomaly_threshold_zscore}")
    logger.info(f"  Worker Interval: {config.worker_interval_seconds}s")
    logger.info(f"  Log Level: {config.log_level}")
    logger.info("=" * 80)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for database to be ready
    wait_for_database()

    # Initialize components
    logger.info("Initializing quality assessment engine...")
    quality_engine.reload_thresholds()
    logger.info("Quality assessment engine initialized")

    # Main processing loop
    cycle_count = 0
    last_threshold_reload = datetime.now()

    logger.info("Entering main worker loop")

    while not shutdown_requested:
        try:
            cycle_count += 1
            cycle_start = time.time()

            logger.info(f"Starting aggregation cycle #{cycle_count}")

            # Run aggregation cycle
            stats = aggregation_service.run_aggregation_cycle()

            cycle_duration = time.time() - cycle_start

            logger.info(
                f"Cycle #{cycle_count} complete in {cycle_duration:.2f}s. "
                f"Windows: {stats['windows_processed']}, "
                f"Insights: {stats['insights_created']}"
            )

            # Reload thresholds periodically (every hour)
            if (datetime.now() - last_threshold_reload).total_seconds() > 3600:
                logger.info("Reloading sensor thresholds from database")
                quality_engine.reload_thresholds()
                last_threshold_reload = datetime.now()

            # Check database health periodically
            if cycle_count % 10 == 0:
                if not check_database_health():
                    logger.error("Database health check failed!")
                    # Wait and retry
                    time.sleep(30)
                    continue

            # Sleep until next cycle
            if not shutdown_requested:
                logger.debug(f"Sleeping for {config.worker_interval_seconds}s until next cycle")
                time.sleep(config.worker_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break

        except Exception as e:
            logger.error(f"Error in worker cycle #{cycle_count}: {e}", exc_info=True)
            # Sleep before retrying
            time.sleep(config.worker_interval_seconds)

    # Cleanup
    logger.info("Shutting down worker...")
    db_manager.close()
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    try:
        run_worker()
    except Exception as e:
        logger.critical(f"Fatal error in worker: {e}", exc_info=True)
        sys.exit(1)




