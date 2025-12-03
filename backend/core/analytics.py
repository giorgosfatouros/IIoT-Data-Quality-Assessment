import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from schemas.visualization import (
    SummaryStatistics, CorrelationMatrix, TimeSeriesPoint, HistogramBin,
    BoxPlotStats, SeasonalDecomposition, AnomalyPoint, SensorAnalysis,
    VisualizationAnalytics, InvalidReadingPoint, SensorInvalidStats,
    InvalidValuesAnalytics, SensorMetadata
)


def _round_float(value: Optional[float]) -> Optional[float]:
    """
    Round a float value to 2 decimal places.
    
    Args:
        value: Float value to round, or None
        
    Returns:
        Rounded float to 2 decimal places, or None if input is None
        Returns 0.0 for NaN or Inf values
    """
    if value is None:
        return None
    
    if not isinstance(value, (int, float)):
        return value
    
    if pd.isna(value) or not np.isfinite(value):
        return 0.0
    
    return round(float(value), 2)

class VisualizationAnalyzer:
    """Core analytics engine for visualization data processing"""
    
    def __init__(self):
        self.rolling_window = 24
        self.anomaly_threshold = 2.0
        self.seasonal_period = 24
    
    def analyze_sensors(
        self, 
        df: pd.DataFrame, 
        selected_columns: List[str],
        table_name: str,
        **kwargs
    ) -> VisualizationAnalytics:
        """
        Comprehensive analysis of selected sensor columns
        """
        # Update parameters
        self.rolling_window = kwargs.get('rolling_window', 24)
        self.anomaly_threshold = kwargs.get('anomaly_threshold', 2.0)
        self.seasonal_period = kwargs.get('seasonal_period', 24)
        
        # Filter data for selected columns
        available_columns = [col for col in selected_columns if col in df.columns]
        if not available_columns:
            raise ValueError("No valid columns found in the dataset")
        
        # Include timestamp column if it exists
        columns_to_include = available_columns.copy()
        if 'timestamp' in df.columns:
            columns_to_include.append('timestamp')
        
        filtered_df = df[columns_to_include].copy()
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(filtered_df)
        
        # Analyze each sensor
        sensor_analyses = []
        for column in available_columns:
            if filtered_df[column].dtype in ['float64', 'int64']:
                analysis = self._analyze_single_sensor(filtered_df, column)
                sensor_analyses.append(analysis)
        
        # Processing information
        processing_info = {
            "total_rows": len(df),
            "selected_columns_count": len(available_columns),
            "rolling_window": self.rolling_window,
            "anomaly_threshold": self.anomaly_threshold,
            "seasonal_period": self.seasonal_period,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return VisualizationAnalytics(
            table_name=table_name,
            selected_columns=available_columns,
            correlation_matrix=correlation_matrix,
            sensor_analyses=sensor_analyses,
            processing_info=processing_info
        )
    
    def _calculate_correlation_matrix(self, df: pd.DataFrame) -> CorrelationMatrix:
        """Calculate correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Round all correlation values to 2 decimal places
        rounded_data = [[_round_float(float(val)) if pd.notna(val) else 0.0 for val in row] 
                        for row in corr_matrix.values.tolist()]
        
        return CorrelationMatrix(
            columns=corr_matrix.columns.tolist(),
            data=rounded_data
        )
    
    def _analyze_single_sensor(self, df: pd.DataFrame, column: str) -> SensorAnalysis:
        """Comprehensive analysis of a single sensor"""
        series = df[column].dropna()
        
        if len(series) == 0:
            raise ValueError(f"No valid data for column {column}")
        
        return SensorAnalysis(
            sensor_name=column,
            summary_stats=self._calculate_summary_stats(series),
            time_series=self._calculate_time_series_analysis(df, column),
            histogram=self._calculate_histogram_data(series),
            box_plot=self._calculate_box_plot_stats(series),
            seasonal_decomposition=self._calculate_seasonal_decomposition(df, column),
            anomalies=self._detect_anomalies(df, column)
        )
    
    def _calculate_summary_stats(self, series: pd.Series) -> SummaryStatistics:
        """Calculate summary statistics for a series"""
        desc = series.describe()
        
        return SummaryStatistics(
            count=float(desc['count']),
            mean=_round_float(float(desc['mean'])),
            std=_round_float(float(desc['std'])),
            min=_round_float(float(desc['min'])),
            q25=_round_float(float(desc['25%'])),
            q50=_round_float(float(desc['50%'])),
            q75=_round_float(float(desc['75%'])),
            max=_round_float(float(desc['max']))
        )
    
    def _calculate_time_series_analysis(self, df: pd.DataFrame, column: str) -> List[TimeSeriesPoint]:
        """Calculate time series analysis with rolling statistics"""
        series = df[column].copy()
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=self.rolling_window, min_periods=1).mean()
        rolling_std = series.rolling(window=self.rolling_window, min_periods=1).std()
        
        time_series_data = []
        
        # Use actual timestamp column if available, otherwise use index or create sequential timestamps
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']  # Already datetime objects from pandas
            if not isinstance(timestamps.iloc[0], (pd.Timestamp, datetime)):
                timestamps = pd.to_datetime(timestamps)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        else:
            # Create sequential timestamps as fallback
            base_time = datetime.now()
            timestamps = pd.date_range(
                start=base_time, 
                periods=len(df), 
                freq='H'
            )
        
        for i, (timestamp, original, r_mean, r_std) in enumerate(
            zip(timestamps, series, rolling_mean, rolling_std)
        ):
            # Skip if we have too many points (limit to 1000 for performance)
            if len(time_series_data) >= 1000:
                break
                
            time_series_data.append(TimeSeriesPoint(
                timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                original=_round_float(float(original) if pd.notna(original) else None),
                rolling_mean=_round_float(float(r_mean) if pd.notna(r_mean) else None),
                rolling_std=_round_float(float(r_std) if pd.notna(r_std) else None)
            ))
        
        return time_series_data
    
    def _calculate_histogram_data(self, series: pd.Series, bins: int = 30) -> List[HistogramBin]:
        """Calculate histogram data for distribution analysis"""
        counts, bin_edges = np.histogram(series.dropna(), bins=bins)
        
        # Calculate density
        bin_width = bin_edges[1] - bin_edges[0]
        density = counts / (len(series) * bin_width)
        
        histogram_data = []
        for i in range(len(counts)):
            histogram_data.append(HistogramBin(
                bin_start=_round_float(float(bin_edges[i])),
                bin_end=_round_float(float(bin_edges[i + 1])),
                count=int(counts[i]),
                density=_round_float(float(density[i]))
            ))
        
        return histogram_data
    
    def _calculate_box_plot_stats(self, series: pd.Series) -> BoxPlotStats:
        """Calculate box plot statistics"""
        clean_series = series.dropna()
        
        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate outliers using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
        
        return BoxPlotStats(
            min=_round_float(float(clean_series.min())),
            q1=_round_float(float(q1)),
            median=_round_float(float(clean_series.median())),
            q3=_round_float(float(q3)),
            max=_round_float(float(clean_series.max())),
            outliers=[_round_float(float(x)) for x in outliers.tolist()[:100]]  # Limit outliers for performance
        )
    
    def _calculate_seasonal_decomposition(self, df: pd.DataFrame, column: str) -> Optional[SeasonalDecomposition]:
        """Calculate seasonal decomposition if enough data points"""
        series = df[column].dropna()
        
        # Need at least 2 full periods for decomposition
        min_periods = self.seasonal_period * 2
        if len(series) < min_periods:
            return None
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                series, 
                model='additive', 
                period=self.seasonal_period,
                extrapolate_trend='freq'
            )
            
            # Create timestamps using actual timestamp column if available
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'][:len(decomposition.observed)]
                if not isinstance(timestamps.iloc[0], (pd.Timestamp, datetime)):
                    timestamps = pd.to_datetime(timestamps)
            elif isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index[:len(decomposition.observed)]
            else:
                base_time = datetime.now()
                timestamps = pd.date_range(
                    start=base_time, 
                    periods=len(decomposition.observed), 
                    freq='H'
                )
            
            # Limit data points for performance
            max_points = 500
            step = max(1, len(decomposition.observed) // max_points)
            
            return SeasonalDecomposition(
                timestamps=[ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts 
                           for ts in timestamps[::step]],
                observed=[_round_float(float(x)) if pd.notna(x) else 0.0 
                         for x in decomposition.observed.iloc[::step]],
                trend=[_round_float(float(x)) if pd.notna(x) else None 
                      for x in decomposition.trend.iloc[::step]],
                seasonal=[_round_float(float(x)) if pd.notna(x) else None 
                         for x in decomposition.seasonal.iloc[::step]],
                residual=[_round_float(float(x)) if pd.notna(x) else None 
                         for x in decomposition.resid.iloc[::step]]
            )
            
        except Exception as e:
            print(f"Seasonal decomposition failed for {column}: {e}")
            return None
    
    def _detect_anomalies(self, df: pd.DataFrame, column: str) -> List[AnomalyPoint]:
        """Detect anomalies using Z-score method"""
        series = df[column].dropna()
        
        if len(series) == 0:
            return []
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(series))
        anomaly_mask = z_scores > self.anomaly_threshold
        
        anomalies = []
        
        # Create timestamps using actual timestamp column if available
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']
            if not isinstance(timestamps.iloc[0], (pd.Timestamp, datetime)):
                timestamps = pd.to_datetime(timestamps)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        else:
            base_time = datetime.now()
            timestamps = pd.date_range(
                start=base_time, 
                periods=len(df), 
                freq='H'
            )
        
        anomaly_indices = series[anomaly_mask].index
        
        for idx in anomaly_indices:  # Return all anomalies
            if idx < len(timestamps):
                timestamp = timestamps[idx]
                anomalies.append(AnomalyPoint(
                    timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                    value=_round_float(float(series.loc[idx])),
                    z_score=_round_float(float(z_scores[series.index.get_loc(idx)]))
                ))
        
        return anomalies

def create_visualization_analyzer() -> VisualizationAnalyzer:
    """Factory function to create visualization analyzer"""
    return VisualizationAnalyzer()


class InvalidValuesAnalyzer:
    """Core analytics engine for invalid values analysis"""
    
    def __init__(self):
        self.default_threshold = 1
        self.tags_metadata = self._load_tags_metadata()
    
    def _load_tags_metadata(self) -> Dict[str, Dict]:
        """Load tags metadata from CSV file"""
        import os
        try:
            # Try backend/config/tags.csv first
            backend_dir = os.path.dirname(os.path.dirname(__file__))
            tags_path = os.path.join(backend_dir, "config", "tags.csv")
            
            if not os.path.exists(tags_path):
                # Try config/tags.csv
                project_dir = os.path.dirname(backend_dir)
                tags_path = os.path.join(project_dir, "config", "tags.csv")
            
            if os.path.exists(tags_path):
                tags_df = pd.read_csv(tags_path)
                # Normalize column names
                tags_df.columns = [c.strip().lower().replace(" ", "_") for c in tags_df.columns]
                
                # Normalize tag names to lowercase for matching
                if "tag" in tags_df.columns:
                    tags_df["tag"] = tags_df["tag"].str.lower()
                    
                    # Create lookup dict
                    tags_dict = {}
                    for _, row in tags_df.iterrows():
                        tag = row["tag"]
                        tags_dict[tag] = row.to_dict()
                    
                    return tags_dict
            
            return {}
        except Exception as e:
            print(f"Warning: Could not load tags metadata: {e}")
            return {}
    
    def _get_sensor_metadata(self, sensor: str) -> Optional[SensorMetadata]:
        """Get metadata for a sensor"""
        # Try different tag name variations
        sensor_lower = sensor.lower()
        
        # Try direct match
        if sensor_lower in self.tags_metadata:
            meta = self.tags_metadata[sensor_lower]
        # Try with COL prefix removed (e.g., "33VI603" -> might be in tags as "33vi603")
        elif sensor_lower.replace("col", "") in self.tags_metadata:
            meta = self.tags_metadata[sensor_lower.replace("col", "")]
        else:
            return None
        
        # Helper function to safely extract string values (handle NaN)
        def safe_str(value):
            if pd.isna(value):
                return None
            return str(value) if value else None
        
        # Parse thresholds
        low_threshold = None
        high_threshold = None
        
        try:
            if "low_threshold" in meta and not pd.isna(meta["low_threshold"]) and meta["low_threshold"]:
                low_threshold = _round_float(float(meta["low_threshold"]))
        except (ValueError, TypeError):
            pass
        
        try:
            if "high_threshold" in meta and not pd.isna(meta["high_threshold"]) and meta["high_threshold"]:
                high_threshold = _round_float(float(meta["high_threshold"]))
        except (ValueError, TypeError):
            pass
        
        return SensorMetadata(
            tag=sensor,
            description=safe_str(meta.get("tag_description")),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            threshold_type=safe_str(meta.get("threshold_type")),
            aggregation_rule=safe_str(meta.get("aggregation_rule")),
            engineering_units=safe_str(meta.get("engineering_units")),
            category=safe_str(meta.get("category"))
        )
    
    def analyze_invalid_values(
        self,
        df: pd.DataFrame,
        selected_columns: Optional[List[str]],
        table_name: str,
        threshold: int = 60,  # Default: 60 alarms per hour
        expected_readings_per_hour: int = 360,  # 10-second frequency = 360 readings/hour
        **kwargs
    ) -> InvalidValuesAnalytics:
        """
        Comprehensive analysis of invalid/alarm values in sensor data from aggregated_insights table
        
        The aggregated_insights table has a row-based structure where each row represents
        one sensor at one timestamp with columns:
        - sensor_tag: sensor name
        - count_invalid: number of invalid readings (alarms) in that hour
        - count_value: number of valid readings in that hour
        - avg_value: average value for that hour
        
        Args:
            df: DataFrame from aggregated_insights (row-based: sensor_tag, count_invalid, etc.)
            selected_columns: List of sensor tags to analyze (optional filter)
            table_name: Machine group name
            threshold: Minimum alarm count per hour to include sensor in results
            expected_readings_per_hour: Expected readings per hour (360 for 10-sec frequency)
        """
        # Validate required columns
        required_cols = ['sensor_tag', 'count_invalid', 'count_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. This endpoint requires aggregated_insights data.")
        
        # Get available sensors
        available_sensors = df['sensor_tag'].unique().tolist()
        
        # Filter by selected columns if provided
        if selected_columns:
            sensors_to_analyze = [s for s in selected_columns if s in available_sensors]
            if not sensors_to_analyze:
                sensors_to_analyze = available_sensors
        else:
            sensors_to_analyze = available_sensors
        
        # Filter sensors with at least one alarm above threshold
        sensors_with_alarms = []
        for sensor in sensors_to_analyze:
            sensor_df = df[df['sensor_tag'] == sensor]
            if len(sensor_df) > 0 and (sensor_df['count_invalid'].fillna(0) >= threshold).any():
                sensors_with_alarms.append(sensor)
        
        if not sensors_with_alarms:
            # Return empty analysis if no alarms found
            total_hours = len(df[df['sensor_tag'].isin(sensors_to_analyze)]) if sensors_to_analyze else 0
            return InvalidValuesAnalytics(
                table_name=table_name,
                selected_columns=sensors_to_analyze,
                threshold=threshold,
                total_readings=total_hours * expected_readings_per_hour * len(sensors_to_analyze),
                total_alarms=0,
                avg_alarms_per_sensor=_round_float(0.0),
                max_alarms_sensor="N/A",
                max_alarms_count=0,
                sensor_stats=[],
                processing_info={
                    "total_rows": len(df),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "sensors_analyzed": len(sensors_to_analyze),
                    "sensors_with_alarms": 0
                }
            )
        
        # Analyze each sensor with alarms
        sensor_stats = []
        total_alarms = 0
        max_alarms_count = 0
        max_alarms_sensor = ""
        
        for sensor in sensors_with_alarms:
            sensor_df = df[df['sensor_tag'] == sensor].copy()
            stats = self._analyze_single_sensor_invalid_rowbased(
                sensor_df, sensor, threshold, expected_readings_per_hour
            )
            sensor_stats.append(stats)
            total_alarms += stats.total_alarms
            
            if stats.total_alarms > max_alarms_count:
                max_alarms_count = stats.total_alarms
                max_alarms_sensor = sensor
        
        # Calculate overall statistics
        total_hours = len(df[df['sensor_tag'].isin(sensors_with_alarms)])
        total_readings = total_hours * expected_readings_per_hour
        avg_alarms_per_sensor = total_alarms / len(sensors_with_alarms) if sensors_with_alarms else 0.0
        
        # Ensure all values are JSON-compliant
        if not np.isfinite(avg_alarms_per_sensor):
            avg_alarms_per_sensor = 0.0
        
        return InvalidValuesAnalytics(
            table_name=table_name,
            selected_columns=sensors_with_alarms,
            threshold=threshold,
            total_readings=total_readings,
            total_alarms=int(total_alarms),
            avg_alarms_per_sensor=_round_float(float(avg_alarms_per_sensor)),
            max_alarms_sensor=max_alarms_sensor,
            max_alarms_count=int(max_alarms_count),
            sensor_stats=sensor_stats,
            processing_info={
                "total_rows": len(df),
                "analysis_timestamp": datetime.now().isoformat(),
                "sensors_analyzed": len(sensors_to_analyze),
                "sensors_with_alarms": len(sensors_with_alarms),
                "expected_readings_per_hour": expected_readings_per_hour
            }
        )
    
    def _analyze_single_sensor_invalid_rowbased(
        self,
        sensor_df: pd.DataFrame,
        sensor: str,
        threshold: int,
        expected_readings_per_hour: int
    ) -> SensorInvalidStats:
        """Analyze invalid values for a single sensor from row-based aggregated_insights data
        
        Args:
            sensor_df: DataFrame filtered to one sensor (already filtered by sensor_tag)
            sensor: Sensor tag name
            threshold: Minimum alarm count per hour to include in invalid_points
            expected_readings_per_hour: Expected readings per hour (360 for 10-sec frequency)
        """
        # Sort by timestamp
        if 'timestamp' in sensor_df.columns:
            sensor_df = sensor_df.sort_values('timestamp').copy()
            timestamps = pd.to_datetime(sensor_df['timestamp'])
        else:
            timestamps = pd.date_range(start=datetime.now(), periods=len(sensor_df), freq='H')
        
        # Get alarm counts (count_invalid column)
        alarm_series = sensor_df['count_invalid'].fillna(0)
        
        # Get mean values (avg_value column)
        mean_series = sensor_df['avg_value'].fillna(0)
        mean_series = mean_series.replace([np.inf, -np.inf], 0)
        
        # Calculate overall alarm statistics
        total_alarms = int(alarm_series.sum())
        total_readings = len(sensor_df) * expected_readings_per_hour
        alarm_percentage = (total_alarms / total_readings * 100) if total_readings > 0 else 0.0
        
        # Ensure alarm_percentage is JSON-compliant
        if not np.isfinite(alarm_percentage):
            alarm_percentage = 0.0
        alarm_percentage = _round_float(alarm_percentage)
        
        # Create time series data with alarm counts
        time_series_data = []
        for i in range(len(sensor_df)):
            timestamp = timestamps.iloc[i] if isinstance(timestamps, pd.Series) else timestamps[i]
            value = mean_series.iloc[i] if i < len(mean_series) else None
            alarm_count = int(alarm_series.iloc[i]) if i < len(alarm_series) else 0
            
            # Ensure value is JSON-compliant
            if pd.notna(value) and np.isfinite(value):
                float_value = _round_float(float(value))
            else:
                float_value = None
            
            time_series_data.append(TimeSeriesPoint(
                timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                original=float_value,
                rolling_mean=None,
                rolling_std=None,
                alarm_count=alarm_count if alarm_count > 0 else None
            ))
        
        # Identify invalid points (where alarm count >= threshold)
        invalid_points = []
        invalid_mask = alarm_series >= threshold
        
        for idx in sensor_df[invalid_mask].index[:500]:  # Limit to 500 points
            row_idx = sensor_df.index.get_loc(idx)
            if row_idx < len(timestamps):
                timestamp = timestamps.iloc[row_idx] if isinstance(timestamps, pd.Series) else timestamps[row_idx]
                value = mean_series.iloc[row_idx]
                alarm_count = int(alarm_series.iloc[row_idx])
                
                if pd.notna(value) and np.isfinite(value):
                    float_value = _round_float(float(value))
                else:
                    float_value = 0.0
                
                invalid_points.append(InvalidReadingPoint(
                    timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                    value=float_value,
                    alarm_count=alarm_count
                ))
        
        # Get sensor metadata
        metadata = self._get_sensor_metadata(sensor)
        
        return SensorInvalidStats(
            sensor_name=sensor,
            total_alarms=total_alarms,
            total_readings=total_readings,
            alarm_percentage=_round_float(float(alarm_percentage)),
            time_series=time_series_data,
            invalid_points=invalid_points,
            metadata=metadata
        )
    
    def _create_time_series(self, df: pd.DataFrame, series: pd.Series, alarm_series: pd.Series = None) -> List[TimeSeriesPoint]:
        """Create time series data from a series with optional alarm counts"""
        time_series_data = []
        
        # Get timestamps
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']
            if not isinstance(timestamps.iloc[0], (pd.Timestamp, datetime)):
                timestamps = pd.to_datetime(timestamps)
        elif 'TIMESTAMP' in df.columns:
            timestamps = pd.to_datetime(df['TIMESTAMP'])
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        else:
            base_time = datetime.now()
            timestamps = pd.date_range(start=base_time, periods=len(df), freq='H')
        
        # Don't limit points - show complete period
        for i in range(len(series)):
            if i >= len(timestamps):
                break
            
            timestamp = timestamps.iloc[i] if isinstance(timestamps, pd.Series) else timestamps[i]
            value = series.iloc[i] if i < len(series) else None
            
            # Ensure value is JSON-compliant (not NaN or Inf)
            if pd.notna(value) and np.isfinite(value):
                float_value = _round_float(float(value))
            else:
                float_value = None
            
            # Get alarm count if alarm_series provided
            alarm_count = None
            if alarm_series is not None and i < len(alarm_series):
                ac = alarm_series.iloc[i]
                if pd.notna(ac) and ac > 0:
                    alarm_count = int(ac)
            
            time_series_data.append(TimeSeriesPoint(
                timestamp=timestamp if isinstance(timestamp, datetime) else timestamp.to_pydatetime(),
                original=float_value,
                rolling_mean=None,
                rolling_std=None,
                alarm_count=alarm_count
            ))
        
        return time_series_data


def create_invalid_values_analyzer() -> InvalidValuesAnalyzer:
    """Factory function to create invalid values analyzer"""
    return InvalidValuesAnalyzer()


class MissingValuesAnalyzer:
    """Analyzer for missing values in sensor data"""
    
    def analyze_missing_values(
        self,
        df: pd.DataFrame,
        table_name: str,
        selected_columns: Optional[List[str]] = None,
        original_freq_sec: int = 10,  # 10 seconds original frequency
        expected_readings_per_hour: int = 360  # 3600 seconds / 10 seconds = 360 readings per hour
    ) -> Dict[str, Any]:
        """
        Analyze missing values in sensor data from aggregated_insights table.
        
        The aggregated_insights table has a row-based structure where each row represents
        one sensor at one timestamp with columns:
        - sensor_tag: sensor name
        - count_missing: number of missing readings in that hour
        - count_value: number of valid readings in that hour
        - expected_count: expected number of readings per hour
        
        IMPORTANT: Original data is collected every 10 SECONDS.
        Each hour represents 360 expected readings (3600s / 10s = 360).
        
        Args:
            df: DataFrame from aggregated_insights (row-based: sensor_tag, count_missing, etc.)
            table_name: Machine group name
            selected_columns: Optional list of sensor tags to analyze
            original_freq_sec: Original data frequency in seconds (default 10 = 10 seconds)
            expected_readings_per_hour: Expected readings per aggregated hour (default 360)
        
        Returns:
            Dictionary with missing values analytics
        """
        try:
            # Validate required columns
            required_cols = ['sensor_tag', 'count_missing', 'count_value', 'expected_count']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. This endpoint requires aggregated_insights data.")
            
            # Get timestamp column
            timestamp_col = self._get_timestamp_column(df)
            if timestamp_col and timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Get available sensors
            available_sensors = df['sensor_tag'].unique().tolist()
            
            # Filter by selected columns if provided
            if selected_columns:
                sensors_to_analyze = [s for s in selected_columns if s in available_sensors]
                if not sensors_to_analyze:
                    sensors_to_analyze = available_sensors
            else:
                sensors_to_analyze = available_sensors
            
            # Calculate time range
            if timestamp_col and timestamp_col in df.columns:
                df_sorted = df.sort_values(timestamp_col)
                total_duration_sec = (df_sorted[timestamp_col].iloc[-1] - df_sorted[timestamp_col].iloc[0]).total_seconds()
            else:
                # Estimate from number of rows (assuming hourly aggregation)
                total_duration_sec = len(df) * 3600
            
            sensor_stats = []
            total_actual = 0
            total_missing = 0
            total_expected = 0
            
            for sensor in sensors_to_analyze:
                sensor_df = df[df['sensor_tag'] == sensor].copy()
                
                if len(sensor_df) == 0:
                    continue
                
                # Sort by timestamp
                if timestamp_col and timestamp_col in sensor_df.columns:
                    sensor_df = sensor_df.sort_values(timestamp_col)
                
                # Get expected count (use mode if available, otherwise use expected_readings_per_hour)
                if 'expected_count' in sensor_df.columns:
                    expected_mode = sensor_df['expected_count'].mode()
                    if len(expected_mode) > 0 and expected_mode.iloc[0] > 0:
                        sensor_expected_per_hour = int(expected_mode.iloc[0])
                    else:
                        sensor_expected_per_hour = expected_readings_per_hour
                else:
                    sensor_expected_per_hour = expected_readings_per_hour
                
                # Calculate actual readings (sum of count_value)
                actual_readings = int(sensor_df['count_value'].fillna(0).sum())
                
                # Calculate expected readings based on actual time range
                # Use timestamp-based calculation if available, otherwise fall back to row count
                if timestamp_col and timestamp_col in sensor_df.columns and len(sensor_df) > 0:
                    # Calculate actual time span from first to last timestamp
                    first_timestamp = pd.to_datetime(sensor_df[timestamp_col].iloc[0])
                    last_timestamp = pd.to_datetime(sensor_df[timestamp_col].iloc[-1])
                    
                    # Get aggregation interval if available (default to 1 hour)
                    if 'aggregation_interval_seconds' in sensor_df.columns:
                        agg_interval = sensor_df['aggregation_interval_seconds'].mode()
                        if len(agg_interval) > 0 and agg_interval.iloc[0] > 0:
                            interval_seconds = int(agg_interval.iloc[0])
                        else:
                            interval_seconds = 3600  # Default 1 hour
                    else:
                        interval_seconds = 3600  # Default 1 hour
                    
                    # Calculate time span in seconds
                    time_span_seconds = (last_timestamp - first_timestamp).total_seconds()
                    
                    # Add one interval to include the last bucket
                    # This accounts for the fact that the last timestamp represents a full interval
                    total_span_seconds = time_span_seconds + interval_seconds
                    
                    # Convert to hours
                    num_hours = total_span_seconds / 3600.0
                else:
                    # Fallback: assume each row represents 1 hour (original behavior)
                    num_hours = len(sensor_df)
                
                expected_readings = int(num_hours * sensor_expected_per_hour)
                
                # Missing readings (sum of count_missing)
                missing_readings = int(sensor_df['count_missing'].fillna(0).sum())
                
                # Missing percentage
                missing_percentage = (missing_readings / expected_readings * 100) if expected_readings > 0 else 0.0
                
                # Identify missing intervals (where count_missing > 0 or count_value < expected)
                missing_intervals = self._identify_missing_intervals_rowbased(
                    sensor_df, timestamp_col, sensor_expected_per_hour
                )
                
                sensor_stats.append({
                    'sensor_name': sensor,
                    'expected_readings': expected_readings,
                    'actual_readings': actual_readings,
                    'missing_readings': missing_readings,
                    'missing_percentage': _round_float(float(missing_percentage) if np.isfinite(missing_percentage) else 0.0),
                    'missing_intervals': missing_intervals
                })
                
                total_actual += actual_readings
                total_missing += missing_readings
                total_expected += expected_readings
            
            # Calculate overall statistics
            overall_missing_percentage = (total_missing / total_expected * 100) if total_expected > 0 else 0.0
            
            return {
                'table_name': table_name,
                'selected_columns': sensors_to_analyze,
                'total_expected_readings': int(total_expected),
                'total_actual_readings': int(total_actual),
                'total_missing_readings': int(total_missing),
                'overall_missing_percentage': _round_float(float(overall_missing_percentage) if np.isfinite(overall_missing_percentage) else 0.0),
                'sensor_stats': sensor_stats,
                'processing_info': {
                    'rows_analyzed': len(df),
                    'sensors_analyzed': len(sensors_to_analyze),
                    'original_freq_sec': original_freq_sec,
                    'expected_readings_per_hour': expected_readings_per_hour,
                    'time_range_hours': _round_float(total_duration_sec / 3600 if total_duration_sec > 0 else 0)
                }
            }
            
        except Exception as e:
            raise ValueError(f"Error analyzing missing values: {str(e)}")
    
    def _identify_missing_intervals_rowbased(
        self, 
        sensor_df: pd.DataFrame, 
        timestamp_col: Optional[str],
        expected_per_hour: int
    ) -> List[Dict[str, Any]]:
        """
        Identify contiguous intervals where readings are missing from row-based aggregated_insights.
        
        Args:
            sensor_df: DataFrame filtered to one sensor with timestamp and count_missing/count_value columns
            timestamp_col: Name of the timestamp column
            expected_per_hour: Expected number of readings per hour
        
        Returns:
            List of missing interval dictionaries
        """
        try:
            if not timestamp_col or timestamp_col not in sensor_df.columns:
                return []
            
            # Sort by timestamp
            sensor_df = sensor_df.sort_values(timestamp_col).copy()
            timestamps = pd.to_datetime(sensor_df[timestamp_col])
            
            # Find timestamps where count_missing > 0 or count_value < expected (indicating missing data)
            missing_mask = (sensor_df['count_missing'].fillna(0) > 0) | (sensor_df['count_value'].fillna(0) < expected_per_hour)
            missing_timestamps = timestamps[missing_mask]
            
            if len(missing_timestamps) == 0:
                return []
            
            intervals = []
            start = missing_timestamps.iloc[0] if isinstance(missing_timestamps, pd.Series) else missing_timestamps[0]
            end = start
            
            # Group contiguous missing periods
            for i in range(1, len(missing_timestamps)):
                current_ts = missing_timestamps.iloc[i] if isinstance(missing_timestamps, pd.Series) else missing_timestamps[i]
                time_diff = (current_ts - end).total_seconds()
                # If gap is <= 2 hours, consider it part of the same interval
                if time_diff <= 7200:  # 2 hours
                    end = current_ts
                else:
                    # Save the interval
                    duration_hours = (end - start).total_seconds() / 3600
                    intervals.append({
                        'start': start.to_pydatetime() if hasattr(start, 'to_pydatetime') else start,
                        'end': end.to_pydatetime() if hasattr(end, 'to_pydatetime') else end,
                        'duration_hours': _round_float(float(duration_hours))
                    })
                    start = current_ts
                    end = current_ts
            
            # Add the last interval
            duration_hours = (end - start).total_seconds() / 3600
            intervals.append({
                'start': start.to_pydatetime() if hasattr(start, 'to_pydatetime') else start,
                'end': end.to_pydatetime() if hasattr(end, 'to_pydatetime') else end,
                'duration_hours': _round_float(float(duration_hours))
            })
            
            return intervals
            
        except Exception as e:
            print(f"Error identifying missing intervals: {e}")
            return []
    
    def _get_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the timestamp column in the dataframe"""
        timestamp_candidates = ['TIMESTAMP', 'timestamp', 'datetime', 'date', 'time', 'DATE']
        for col in timestamp_candidates:
            if col in df.columns:
                return col
        return None


def create_missing_values_analyzer() -> MissingValuesAnalyzer:
    """Factory function to create missing values analyzer"""
    return MissingValuesAnalyzer()


class DataQualityAnalyzer:
    """Analyzer for comprehensive data quality assessment"""
    
    def analyze_data_quality(
        self,
        df: pd.DataFrame,
        table_name: str,
        tags_df: Optional[pd.DataFrame] = None,
        completeness_threshold: float = 90.0,
        correlation_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Perform comprehensive data quality analysis"""
        
        # Preprocess aggregated data to calculate mean values
        processed_df, sensor_names, raw_aggregates = self._preprocess_aggregated_data(df)
        
        # General Information (use raw aggregates for counts)
        general_info = self._analyze_general_info(df, processed_df, raw_aggregates)
        
        # Descriptive Statistics (use processed mean values)
        descriptive_stats = self._calculate_descriptive_stats(processed_df, raw_aggregates)
        
        # Consistency Checks
        consistency_check = self._check_consistency(df)
        
        # Completeness Check (use raw count columns)
        completeness_check = self._check_completeness(raw_aggregates, completeness_threshold)
        
        # Outliers (use mean values)
        outliers = self._calculate_outliers(processed_df)
        
        # Accuracy Issues (if tags data provided, use mean values)
        accuracy_issues = self._check_accuracy(processed_df, tags_df) if tags_df is not None else []
        
        # Correlation Analysis (use mean values)
        correlation_matrix, strong_correlations = self._analyze_correlations(processed_df, correlation_threshold)
        
        return {
            "table_name": table_name,
            "general_info": general_info,
            "descriptive_stats": descriptive_stats,
            "consistency_check": consistency_check,
            "completeness_check": completeness_check,
            "outliers": outliers,
            "accuracy_issues": accuracy_issues,
            "strong_correlations": strong_correlations,
            "correlation_matrix": correlation_matrix
        }
    
    def _preprocess_aggregated_data(self, df: pd.DataFrame) -> tuple:
        """
        Preprocess aggregated hourly data from row-based aggregated_insights structure.
        
        The aggregated_insights table has a row-based structure where each row represents
        one sensor at one timestamp with columns:
        - sensor_tag: sensor name
        - avg_value: average value for that hour
        - count_value: number of valid readings
        - sum_value: sum of values
        - min_value, max_value: min/max values
        
        Returns:
            - processed_df: DataFrame with mean values per sensor (pivoted: timestamp as index, sensors as columns)
            - sensor_names: List of sensor names
            - raw_aggregates: Dict with original data organized by sensor
        """
        # Validate required columns
        required_cols = ['sensor_tag', 'avg_value', 'count_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. This analyzer requires aggregated_insights data.")
        
        # Get timestamp column
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'TIMESTAMP'
        if timestamp_col not in df.columns:
            raise ValueError("Missing timestamp column. Required for data quality analysis.")
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Get unique sensors
        sensors = df['sensor_tag'].unique().tolist()
        
        # Pivot to have sensors as columns (for correlation analysis)
        processed_df = df.pivot(index=timestamp_col, columns='sensor_tag', values='avg_value')
        
        # Store raw aggregates organized by sensor
        raw_aggregates = {
            'sum': {},
            'count': {},
            'min': {},
            'max': {}
        }
        
        for sensor in sensors:
            sensor_df = df[df['sensor_tag'] == sensor].copy()
            sensor_df = sensor_df.set_index(timestamp_col).sort_index()
            
            # Store raw aggregates
            if 'sum_value' in sensor_df.columns:
                raw_aggregates['sum'][sensor] = sensor_df['sum_value']
            if 'count_value' in sensor_df.columns:
                raw_aggregates['count'][sensor] = sensor_df['count_value']
            if 'min_value' in sensor_df.columns:
                raw_aggregates['min'][sensor] = sensor_df['min_value']
            if 'max_value' in sensor_df.columns:
                raw_aggregates['max'][sensor] = sensor_df['max_value']
        
        # Convert raw aggregates to DataFrames (align with processed_df index)
        for key in raw_aggregates:
            if raw_aggregates[key]:
                # Create DataFrame with same index as processed_df
                raw_df = pd.DataFrame(index=processed_df.index)
                for sensor in sensors:
                    if sensor in raw_aggregates[key]:
                        raw_df[sensor] = raw_aggregates[key][sensor]
                raw_aggregates[key] = raw_df
        
        return processed_df, sensors, raw_aggregates
    
    def _analyze_general_info(self, df: pd.DataFrame, processed_df: pd.DataFrame, raw_aggregates: Dict) -> Dict[str, Any]:
        """Analyze general dataset information"""
        
        # Date range (from processed_df index which is timestamp)
        date_range_start = None
        date_range_end = None
        
        if isinstance(processed_df.index, pd.DatetimeIndex) and not processed_df.index.empty:
            date_range_start = processed_df.index.min().isoformat()
            date_range_end = processed_df.index.max().isoformat()
        elif isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
            date_range_start = df.index.min().isoformat()
            date_range_end = df.index.max().isoformat()
        elif 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'])
                date_range_start = timestamps.min().isoformat()
                date_range_end = timestamps.max().isoformat()
            except:
                pass
        elif 'TIMESTAMP' in df.columns:
            try:
                timestamps = pd.to_datetime(df['TIMESTAMP'])
                date_range_start = timestamps.min().isoformat()
                date_range_end = timestamps.max().isoformat()
            except:
                pass
        
        # Total data points and missing values (use processed mean values)
        total_data_points = processed_df.size
        total_missing_values = int(processed_df.isnull().sum().sum())
        missing_percentage = (total_missing_values / total_data_points * 100) if total_data_points > 0 else 0.0
        
        # Data points per sensor (use count from raw aggregates if available)
        sensor_data_points = []
        for col in processed_df.columns:
            # Try to get actual data points from COUNT column
            if raw_aggregates.get('count') is not None and col in raw_aggregates['count']:
                data_points = int(raw_aggregates['count'][col].sum())
            else:
                data_points = int(processed_df[col].notnull().sum())
            
            missing_pct = processed_df[col].isnull().mean() * 100
            sensor_data_points.append({
                "sensor_name": col,
                "data_points": data_points,
                "missing_percentage": _round_float(float(missing_pct))
            })
        
        return {
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "total_data_points": int(total_data_points),
            "total_missing_values": total_missing_values,
            "missing_percentage": _round_float(float(missing_percentage)),
            "num_sensors": len(processed_df.columns),
            "sensor_data_points": sensor_data_points
        }
    
    def _calculate_descriptive_stats(self, numeric_df: pd.DataFrame, raw_aggregates: Dict) -> List[Dict[str, Any]]:
        """Calculate descriptive statistics for each sensor"""
        stats = []
        desc = numeric_df.describe()
        
        for col in numeric_df.columns:
            if col in desc.columns:
                # Use actual MIN/MAX from raw aggregates if available
                min_val = float(desc.loc['min', col]) if pd.notna(desc.loc['min', col]) else 0.0
                max_val = float(desc.loc['max', col]) if pd.notna(desc.loc['max', col]) else 0.0
                
                if raw_aggregates.get('min') is not None and col in raw_aggregates['min']:
                    min_val = float(raw_aggregates['min'][col].min())
                if raw_aggregates.get('max') is not None and col in raw_aggregates['max']:
                    max_val = float(raw_aggregates['max'][col].max())
                
                stats.append({
                    "sensor_name": col,
                    "count": float(desc.loc['count', col]),
                    "mean": _round_float(float(desc.loc['mean', col]) if pd.notna(desc.loc['mean', col]) else 0.0),
                    "std": _round_float(float(desc.loc['std', col]) if pd.notna(desc.loc['std', col]) else 0.0),
                    "min": _round_float(min_val),
                    "q25": _round_float(float(desc.loc['25%', col]) if pd.notna(desc.loc['25%', col]) else 0.0),
                    "q50": _round_float(float(desc.loc['50%', col]) if pd.notna(desc.loc['50%', col]) else 0.0),
                    "q75": _round_float(float(desc.loc['75%', col]) if pd.notna(desc.loc['75%', col]) else 0.0),
                    "max": _round_float(max_val)
                })
        
        return stats
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency (duplicates and timestamp order)"""
        
        # Check for duplicates
        duplicate_rows = df[df.duplicated(keep=False)]
        duplicate_count = len(duplicate_rows) // 2  # Divide by 2 since duplicates are counted twice
        duplicate_percentage = (duplicate_count / len(df) * 100) if len(df) > 0 else 0.0
        
        # Check timestamp consistency
        timestamps_consistent = True
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps_consistent = df.index.is_monotonic_increasing
        elif 'TIMESTAMP' in df.columns:
            try:
                timestamps = pd.to_datetime(df['TIMESTAMP'])
                timestamps_consistent = timestamps.is_monotonic_increasing
            except:
                pass
        
        return {
            "has_duplicates": duplicate_count > 0,
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": _round_float(float(duplicate_percentage)),
            "timestamps_consistent": timestamps_consistent
        }
    
    def _check_completeness(self, raw_aggregates: Dict, threshold: float) -> Dict[str, Any]:
        """Check data completeness using COUNT columns"""
        
        # Use count data if available, otherwise fallback to mean values
        if raw_aggregates.get('count') is not None and len(raw_aggregates['count']) > 0:
            count_df = raw_aggregates['count']
            # Calculate completeness: hours with data / total hours
            completeness_pct = (count_df > 0).mean() * 100
            overall_completeness = float(completeness_pct.mean())
            
            # Find columns below threshold
            incomplete_sensors = []
            for col in count_df.columns:
                col_completeness = completeness_pct[col]
                if col_completeness < threshold:
                    incomplete_sensors.append({
                        "sensor_name": col,
                        "completeness": _round_float(float(col_completeness))
                    })
        else:
            # Fallback if no count data available
            overall_completeness = 100.0
            incomplete_sensors = []
        
        return {
            "overall_completeness": _round_float(overall_completeness),
            "completeness_threshold": threshold,
            "incomplete_sensors": incomplete_sensors
        }
    
    def _calculate_outliers(self, numeric_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate outliers using IQR method"""
        outliers_list = []
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_condition = (numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))
            outlier_percentage = outlier_condition.mean() * 100
            
            outliers_list.append({
                "sensor_name": col,
                "outlier_percentage": _round_float(float(outlier_percentage))
            })
        
        return outliers_list
    
    def _check_accuracy(self, numeric_df: pd.DataFrame, tags_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check data accuracy against thresholds from tags"""
        accuracy_issues = []
        
        # Prepare tags data
        tags_df_copy = tags_df.copy()
        if 'tag' in tags_df_copy.columns:
            tags_df_copy['tag'] = tags_df_copy['tag'].str.lower()
            tags_df_copy = tags_df_copy.set_index('tag')
        
        for col in numeric_df.columns:
            # Column names in processed_df are already clean sensor names (e.g., "33VI603")
            tag_name = col.lower()
            
            if tag_name not in tags_df_copy.index:
                continue
            
            sensor_data = numeric_df[col].dropna()
            if sensor_data.empty:
                continue
            
            tag_row = tags_df_copy.loc[tag_name]
            
            # Get thresholds
            low_threshold = tag_row.get('low_threshold')
            high_threshold = tag_row.get('high_threshold')
            threshold_type = tag_row.get('threshold_type')
            
            # Check for issues
            issues = pd.Series([False] * len(sensor_data), index=sensor_data.index)
            
            if pd.notna(low_threshold) and threshold_type in ['Down', 'Up/Down']:
                issues |= (sensor_data < float(low_threshold))
            
            if pd.notna(high_threshold) and threshold_type in ['Up', 'Up/Down']:
                issues |= (sensor_data > float(high_threshold))
            
            issues_percentage = issues.sum() / len(sensor_data) * 100
            
            if issues_percentage > 0:
                accuracy_issues.append({
                    "sensor_name": col,
                    "issues_percentage": _round_float(float(issues_percentage)),
                    "threshold_type": threshold_type if pd.notna(threshold_type) else None,
                    "low_threshold": _round_float(float(low_threshold) if pd.notna(low_threshold) else None),
                    "high_threshold": _round_float(float(high_threshold) if pd.notna(high_threshold) else None)
                })
        
        return accuracy_issues
    
    def _analyze_correlations(self, numeric_df: pd.DataFrame, threshold: float) -> tuple:
        """Analyze correlations between sensors"""
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Replace NaN/inf with None for JSON serialization
        corr_matrix_clean = corr_matrix.fillna(0)
        
        # Convert to dict format with rounded values
        rounded_data = [[_round_float(float(val)) if pd.notna(val) else 0.0 for val in row] 
                        for row in corr_matrix_clean.values.tolist()]
        correlation_dict = {
            "columns": corr_matrix_clean.columns.tolist(),
            "data": rounded_data
        }
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if pd.notna(corr_value) and abs(corr_value) > threshold:
                    strong_correlations.append({
                        "sensor_a": corr_matrix.columns[i],
                        "sensor_b": corr_matrix.columns[j],
                        "correlation": _round_float(float(corr_value))
                    })
        
        return correlation_dict, strong_correlations


def create_data_quality_analyzer() -> DataQualityAnalyzer:
    """Factory function to create data quality analyzer"""
    return DataQualityAnalyzer()
