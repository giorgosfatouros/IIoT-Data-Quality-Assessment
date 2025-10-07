from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from enum import Enum

class DataSourceType(str, Enum):
    RAW = "raw"
    AGGREGATED = "aggregated"
    AUTO = "auto"  # Automatically detect based on table name

class SummaryStatistics(BaseModel):
    count: float
    mean: float
    std: float
    min: float
    q25: float
    q50: float
    q75: float
    max: float

class CorrelationMatrix(BaseModel):
    columns: List[str]
    data: List[List[float]]

class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    original: Optional[float]
    rolling_mean: Optional[float]
    rolling_std: Optional[float]
    alarm_count: Optional[int] = None  # Number of alarms at this timestamp

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    density: float

class BoxPlotStats(BaseModel):
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers: List[float]

class SeasonalDecomposition(BaseModel):
    timestamps: List[datetime]
    observed: List[float]
    trend: List[Optional[float]]
    seasonal: List[Optional[float]]
    residual: List[Optional[float]]

class AnomalyPoint(BaseModel):
    timestamp: datetime
    value: float
    z_score: float

class SensorAnalysis(BaseModel):
    sensor_name: str
    summary_stats: SummaryStatistics
    time_series: List[TimeSeriesPoint]
    histogram: List[HistogramBin]
    box_plot: BoxPlotStats
    seasonal_decomposition: Optional[SeasonalDecomposition]
    anomalies: List[AnomalyPoint]

class VisualizationAnalytics(BaseModel):
    table_name: str
    selected_columns: List[str]
    correlation_matrix: CorrelationMatrix
    sensor_analyses: List[SensorAnalysis]
    processing_info: Dict[str, Any]

class VisualizationRequest(BaseModel):
    table: str
    columns: List[str]
    limit: Optional[int] = 5000
    rolling_window: Optional[int] = 24
    anomaly_threshold: Optional[float] = 2.0
    seasonal_period: Optional[int] = 24
    data_source: Optional[DataSourceType] = DataSourceType.AUTO
    date_from: Optional[date] = None
    date_to: Optional[date] = None

# Invalid Values Analysis Schemas
class InvalidReadingPoint(BaseModel):
    timestamp: datetime
    value: float
    alarm_count: int

class SensorMetadata(BaseModel):
    tag: str
    description: Optional[str] = None
    low_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
    threshold_type: Optional[str] = None
    aggregation_rule: Optional[str] = None
    engineering_units: Optional[str] = None
    category: Optional[str] = None

class SensorInvalidStats(BaseModel):
    sensor_name: str
    total_alarms: int
    total_readings: int
    alarm_percentage: float
    time_series: List[TimeSeriesPoint]
    invalid_points: List[InvalidReadingPoint]
    metadata: Optional[SensorMetadata] = None

class InvalidValuesAnalytics(BaseModel):
    table_name: str
    selected_columns: List[str]
    threshold: int
    total_readings: int
    total_alarms: int
    avg_alarms_per_sensor: float
    max_alarms_sensor: str
    max_alarms_count: int
    sensor_stats: List[SensorInvalidStats]
    processing_info: Dict[str, Any]

class InvalidValuesRequest(BaseModel):
    table: str
    columns: Optional[List[str]] = None  # If None, analyze all sensors
    threshold: Optional[int] = 60
    limit: Optional[int] = 5000
    date_from: Optional[date] = None
    date_to: Optional[date] = None

# Missing Values Analysis Models
class MissingInterval(BaseModel):
    start: datetime
    end: datetime
    duration_hours: float

class SensorMissingStats(BaseModel):
    sensor_name: str
    expected_readings: int
    actual_readings: int
    missing_readings: int
    missing_percentage: float
    missing_intervals: List[MissingInterval]

class MissingValuesAnalytics(BaseModel):
    table_name: str
    selected_columns: List[str]
    total_expected_readings: int
    total_actual_readings: int
    total_missing_readings: int
    overall_missing_percentage: float
    sensor_stats: List[SensorMissingStats]
    processing_info: Dict[str, Any]

class MissingValuesRequest(BaseModel):
    table: str
    columns: Optional[List[str]] = None  # If None, analyze all sensors
    limit: Optional[int] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None

# Data Quality Assessment Models
class SensorDataPoints(BaseModel):
    sensor_name: str
    data_points: int
    missing_percentage: float

class GeneralInfo(BaseModel):
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    total_data_points: int
    total_missing_values: int
    missing_percentage: float
    num_sensors: int
    sensor_data_points: List[SensorDataPoints]

class DescriptiveStats(BaseModel):
    sensor_name: str
    count: float
    mean: float
    std: float
    min: float
    q25: float
    q50: float
    q75: float
    max: float

class ConsistencyCheck(BaseModel):
    has_duplicates: bool
    duplicate_count: int
    duplicate_percentage: float
    timestamps_consistent: bool

class CompletenessCheck(BaseModel):
    overall_completeness: float
    completeness_threshold: float
    incomplete_sensors: List[Dict[str, float]]  # {sensor_name: completeness_percentage}

class OutlierInfo(BaseModel):
    sensor_name: str
    outlier_percentage: float

class AccuracyIssue(BaseModel):
    sensor_name: str
    issues_percentage: float
    threshold_type: Optional[str] = None
    low_threshold: Optional[float] = None
    high_threshold: Optional[float] = None

class CorrelationPair(BaseModel):
    sensor_a: str
    sensor_b: str
    correlation: float

class DataQualityAnalytics(BaseModel):
    table_name: str
    general_info: GeneralInfo
    descriptive_stats: List[DescriptiveStats]
    consistency_check: ConsistencyCheck
    completeness_check: CompletenessCheck
    outliers: List[OutlierInfo]
    accuracy_issues: List[AccuracyIssue]
    strong_correlations: List[CorrelationPair]
    correlation_matrix: Optional[Dict[str, Any]] = None

class DataQualityRequest(BaseModel):
    table: str
    limit: Optional[int] = 5000
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    completeness_threshold: float = 90.0
    correlation_threshold: float = 0.7
