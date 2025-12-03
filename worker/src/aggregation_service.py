"""
Aggregation Service
Processes raw sensor data and generates aggregated insights
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import text

from worker.src.config import config
from worker.src.database import db_manager
from worker.src.quality_assessment import quality_engine

logger = logging.getLogger(__name__)


class AggregationService:
    """Handles aggregation of raw sensor data into quality insights"""

    def __init__(self):
        """Initialize aggregation service"""
        self.aggregation_interval = timedelta(seconds=config.aggregation_interval_seconds)

    def get_last_aggregation_timestamp(self) -> Optional[datetime]:
        """
        Get the timestamp of the last processed aggregation

        Returns:
            Datetime of last aggregation or None if no aggregations exist
        """
        try:
            query = """
            SELECT MAX(timestamp) as last_timestamp
            FROM aggregated_insights
            """
            result = db_manager.execute_query(query)

            if result and result[0][0]:
                return result[0][0]
            return None

        except Exception as e:
            logger.error(f"Failed to get last aggregation timestamp: {e}")
            return None

    def get_pending_aggregation_windows(self) -> List[datetime]:
        """
        Get list of time windows that need aggregation

        Returns:
            List of window start timestamps
        """
        try:
            # Get the latest raw data timestamp
            query_latest = """
            SELECT MAX(timestamp) as latest_timestamp
            FROM raw_sensor_data
            """
            result = db_manager.execute_query(query_latest)

            if not result or not result[0][0]:
                logger.info("No raw data available for aggregation")
                return []

            latest_raw_timestamp = result[0][0]

            # Get last aggregation timestamp
            last_aggregation = self.get_last_aggregation_timestamp()

            # If no aggregations exist, start from the earliest raw data
            if not last_aggregation:
                query_earliest = """
                SELECT MIN(timestamp) as earliest_timestamp
                FROM raw_sensor_data
                """
                result = db_manager.execute_query(query_earliest)
                start_time = result[0][0] if result and result[0][0] else latest_raw_timestamp
            else:
                start_time = last_aggregation + self.aggregation_interval

            # Generate list of pending windows
            windows = []
            current_window = start_time.replace(minute=0, second=0, microsecond=0)

            while current_window + self.aggregation_interval <= latest_raw_timestamp:
                windows.append(current_window)
                current_window += self.aggregation_interval

            logger.info(f"Found {len(windows)} pending aggregation windows")
            return windows

        except Exception as e:
            logger.error(f"Failed to get pending aggregation windows: {e}")
            return []

    def aggregate_window(self, window_start: datetime) -> int:
        """
        Aggregate raw data for a specific time window

        Args:
            window_start: Start timestamp of the aggregation window

        Returns:
            Number of insights created
        """
        window_end = window_start + self.aggregation_interval

        try:
            # Fetch raw data for this window
            query = """
            SELECT timestamp, sensor_tag, value, machine_group, quality_flag
            FROM raw_sensor_data
            WHERE timestamp >= :window_start AND timestamp < :window_end
            ORDER BY timestamp, sensor_tag
            """
            
            results = db_manager.execute_query(query, {
                'window_start': window_start,
                'window_end': window_end
            })

            if not results:
                logger.debug(f"No data found for window {window_start}")
                return 0

            # Convert to DataFrame
            df = pd.DataFrame(results, columns=['timestamp', 'sensor_tag', 'value', 'machine_group', 'quality_flag'])

            # Group by sensor and machine
            insights_created = 0
            for (sensor_tag, machine_group), group_df in df.groupby(['sensor_tag', 'machine_group']):
                try:
                    insight = self._create_insight(group_df, sensor_tag, machine_group, window_start, window_end)
                    if insight:
                        self._insert_insight(insight)
                        insights_created += 1
                except Exception as e:
                    logger.error(f"Failed to create insight for {sensor_tag}/{machine_group}: {e}")
                    continue

            return insights_created

        except Exception as e:
            logger.error(f"Failed to aggregate window {window_start}: {e}")
            return 0

    def _create_insight(
        self,
        df: pd.DataFrame,
        sensor_tag: str,
        machine_group: str,
        window_start: datetime,
        window_end: datetime
    ) -> Optional[Dict]:
        """Create an aggregated insight from raw data"""
        if df.empty:
            return None

        values = df['value'].dropna()
        if len(values) == 0:
            return None

        # Calculate statistics
        sum_value = float(values.sum())
        count_value = len(values)
        min_value = float(values.min())
        max_value = float(values.max())
        avg_value = float(values.mean())
        stddev_value = float(values.std()) if len(values) > 1 else 0.0

        # Count quality issues
        count_invalid = int((df['quality_flag'] == 'invalid').sum())
        count_missing = 0  # Will be calculated based on expected vs actual
        count_anomaly = int((df['quality_flag'] == 'anomaly').sum())

        # Calculate expected readings
        expected_count = config.expected_readings_per_interval
        count_missing = max(0, expected_count - count_value)

        # Calculate quality scores
        quality_scores = quality_engine.calculate_quality_scores(
            total_readings=count_value,
            valid_readings=count_value - count_invalid,
            missing_readings=count_missing,
            anomaly_readings=count_anomaly,
            expected_readings=expected_count
        )

        return {
            'timestamp': window_start,
            'machine_group': machine_group,
            'sensor_tag': sensor_tag,
            'sum_value': sum_value,
            'count_value': count_value,
            'min_value': min_value,
            'max_value': max_value,
            'avg_value': avg_value,
            'stddev_value': stddev_value,
            'count_invalid': count_invalid,
            'count_missing': count_missing,
            'count_anomaly': count_anomaly,
            'completeness_score': quality_scores['completeness_score'],
            'validity_score': quality_scores['validity_score'],
            'anomaly_score': quality_scores['anomaly_score'],
            'overall_quality_score': quality_scores['overall_quality_score'],
            'aggregation_interval_seconds': config.aggregation_interval_seconds,
            'expected_count': expected_count
        }

    def _insert_insight(self, insight: Dict):
        """Insert or update an aggregated insight"""
        query = """
        INSERT INTO aggregated_insights (
            timestamp, machine_group, sensor_tag,
            sum_value, count_value, min_value, max_value, avg_value, stddev_value,
            count_invalid, count_missing, count_anomaly,
            completeness_score, validity_score, anomaly_score, overall_quality_score,
            aggregation_interval_seconds, expected_count
        ) VALUES (
            :timestamp, :machine_group, :sensor_tag,
            :sum_value, :count_value, :min_value, :max_value, :avg_value, :stddev_value,
            :count_invalid, :count_missing, :count_anomaly,
            :completeness_score, :validity_score, :anomaly_score, :overall_quality_score,
            :aggregation_interval_seconds, :expected_count
        )
        ON CONFLICT (timestamp, sensor_tag, machine_group, aggregation_interval_seconds)
        DO UPDATE SET
            sum_value = EXCLUDED.sum_value,
            count_value = EXCLUDED.count_value,
            min_value = EXCLUDED.min_value,
            max_value = EXCLUDED.max_value,
            avg_value = EXCLUDED.avg_value,
            stddev_value = EXCLUDED.stddev_value,
            count_invalid = EXCLUDED.count_invalid,
            count_missing = EXCLUDED.count_missing,
            count_anomaly = EXCLUDED.count_anomaly,
            completeness_score = EXCLUDED.completeness_score,
            validity_score = EXCLUDED.validity_score,
            anomaly_score = EXCLUDED.anomaly_score,
            overall_quality_score = EXCLUDED.overall_quality_score,
            expected_count = EXCLUDED.expected_count
        """
        
        db_manager.execute_update(query, insight)

    def run_aggregation_cycle(self) -> Dict[str, int]:
        """
        Run a complete aggregation cycle for all pending windows

        Returns:
            Dictionary with statistics about the cycle
        """
        windows = self.get_pending_aggregation_windows()
        
        if not windows:
            logger.debug("No pending aggregation windows")
            return {
                'windows_processed': 0,
                'insights_created': 0
            }

        insights_created = 0
        for window in windows:
            try:
                count = self.aggregate_window(window)
                insights_created += count
            except Exception as e:
                logger.error(f"Failed to aggregate window {window}: {e}")
                continue

        return {
            'windows_processed': len(windows),
            'insights_created': insights_created
        }


# Global aggregation service instance
aggregation_service = AggregationService()




