"""
Quality Assessment Engine
Validates sensor readings against thresholds and detects anomalies
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

from worker.src.config import config
from worker.src.database import db_manager

logger = logging.getLogger(__name__)


class QualityAssessmentEngine:
    """Engine for assessing data quality based on thresholds and statistical methods"""

    def __init__(self):
        """Initialize quality assessment engine"""
        self.thresholds_cache: Dict[str, Dict] = {}
        self.last_reload = None

    def reload_thresholds(self):
        """Reload sensor thresholds from database"""
        try:
            query = """
                SELECT tag, low_threshold, high_threshold, threshold_type, aggregation_rule
                FROM sensor_thresholds
            """
            results = db_manager.execute_query(query)
            
            self.thresholds_cache = {}
            for row in results:
                tag, low, high, threshold_type, agg_rule = row
                self.thresholds_cache[tag] = {
                    'low_threshold': float(low) if low is not None else None,
                    'high_threshold': float(high) if high is not None else None,
                    'threshold_type': threshold_type,
                    'aggregation_rule': agg_rule
                }
            
            self.last_reload = datetime.now()
            logger.info(f"Loaded {len(self.thresholds_cache)} sensor thresholds")
        except Exception as e:
            logger.error(f"Failed to reload thresholds: {e}")
            raise

    def get_threshold(self, sensor_tag: str) -> Optional[Dict]:
        """Get threshold configuration for a sensor"""
        if not self.thresholds_cache:
            self.reload_thresholds()
        return self.thresholds_cache.get(sensor_tag)

    def validate_value(self, sensor_tag: str, value: float) -> Tuple[bool, str]:
        """
        Validate a sensor reading against its thresholds
        
        Returns:
            (is_valid, quality_flag) where quality_flag is 'valid' or 'invalid'
        """
        threshold = self.get_threshold(sensor_tag)
        if not threshold:
            # No threshold defined, consider valid
            return True, 'valid'

        low = threshold['low_threshold']
        high = threshold['high_threshold']
        threshold_type = threshold['threshold_type']

        is_valid = True

        if threshold_type == 'Down' and low is not None:
            is_valid = value >= low
        elif threshold_type == 'Up' and high is not None:
            is_valid = value <= high
        elif threshold_type == 'Up/Down':
            if low is not None and value < low:
                is_valid = False
            if high is not None and value > high:
                is_valid = False

        quality_flag = 'valid' if is_valid else 'invalid'
        return is_valid, quality_flag

    def detect_anomaly(self, values: pd.Series) -> pd.Series:
        """
        Detect statistical anomalies using Z-score method
        
        Args:
            values: Series of sensor readings
            
        Returns:
            Series of boolean values indicating anomalies
        """
        if len(values) < 2:
            return pd.Series([False] * len(values), index=values.index)

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(values.dropna()))
        
        # Create anomaly mask
        anomaly_mask = pd.Series([False] * len(values), index=values.index)
        valid_indices = values.dropna().index
        anomaly_mask.loc[valid_indices] = z_scores > config.anomaly_threshold_zscore
        
        return anomaly_mask

    def calculate_quality_scores(
        self,
        total_readings: int,
        valid_readings: int,
        missing_readings: int,
        anomaly_readings: int,
        expected_readings: int
    ) -> Dict[str, float]:
        """
        Calculate quality scores from reading counts
        
        Returns:
            Dictionary with completeness_score, validity_score, anomaly_score, overall_quality_score
        """
        # Completeness: % of expected readings present
        completeness_score = ((expected_readings - missing_readings) / expected_readings * 100) if expected_readings > 0 else 0.0
        
        # Validity: % of readings within thresholds
        validity_score = (valid_readings / total_readings * 100) if total_readings > 0 else 0.0
        
        # Anomaly: % of readings that are NOT anomalies
        non_anomaly_readings = total_readings - anomaly_readings
        anomaly_score = (non_anomaly_readings / total_readings * 100) if total_readings > 0 else 0.0
        
        # Overall: Weighted composite (Completeness 30%, Validity 50%, Anomaly 20%)
        overall_quality_score = (
            (completeness_score * 0.30) +
            (validity_score * 0.50) +
            (anomaly_score * 0.20)
        )
        
        # Ensure scores are within 0-100 range
        completeness_score = max(0.0, min(100.0, completeness_score))
        validity_score = max(0.0, min(100.0, validity_score))
        anomaly_score = max(0.0, min(100.0, anomaly_score))
        overall_quality_score = max(0.0, min(100.0, overall_quality_score))
        
        return {
            'completeness_score': completeness_score,
            'validity_score': validity_score,
            'anomaly_score': anomaly_score,
            'overall_quality_score': overall_quality_score
        }


# Global quality engine instance
quality_engine = QualityAssessmentEngine()




