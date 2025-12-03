from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
import os
from datetime import date

from schemas.visualization import (
    VisualizationAnalytics, VisualizationRequest, DataSourceType,
    InvalidValuesAnalytics, InvalidValuesRequest,
    MissingValuesAnalytics, MissingValuesRequest,
    DataQualityAnalytics, DataQualityRequest
)
from core.analytics import create_visualization_analyzer, create_invalid_values_analyzer, create_missing_values_analyzer, create_data_quality_analyzer
from models.database import get_engine

router = APIRouter(prefix="/analytics", tags=["visualization"])

def _detect_data_source_type(table: str) -> DataSourceType:
    """Detect data source type - always aggregated for TimescaleDB (aggregated_insights)"""
    # With TimescaleDB, we always use aggregated_insights table
    return DataSourceType.AGGREGATED

def _get_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Find the timestamp column in the dataframe"""
    timestamp_candidates = ['timestamp', 'TIMESTAMP', 'time', 'TIME', 'date', 'DATE', 'datetime', 'DATETIME']
    for col in timestamp_candidates:
        if col in df.columns:
            return col
    return None

def _fetch_table_dataframe(
    table: str, 
    limit: Optional[int] = 5000, 
    date_from: Optional[date] = None, 
    date_to: Optional[date] = None,
    sensor_tags: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch data from aggregated_insights table and return as DataFrame with optional filtering
    
    Args:
        table: Machine group name (not table name)
        limit: Maximum number of rows to fetch
        date_from: Start date for filtering
        date_to: End date for filtering
        sensor_tags: Optional list of sensor tags to filter
    """
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        # Build query
        where_conditions = ["machine_group = :machine_group"]
        params = {"machine_group": machine_group}
        
        # Add date filtering
        if date_from:
            where_conditions.append("timestamp >= :date_from")
            params["date_from"] = date_from
        if date_to:
            # Add one day to include the entire end date
            end_date = pd.to_datetime(date_to) + pd.Timedelta(days=1)
            where_conditions.append("timestamp < :date_to")
            params["date_to"] = end_date.strftime("%Y-%m-%d")
        
        # Add sensor tag filtering
        if sensor_tags:
            where_conditions.append("sensor_tag = ANY(:sensor_tags)")
            params["sensor_tags"] = sensor_tags
        
        # Construct query
        where_clause = " AND ".join(where_conditions)
        limit_clause = f" LIMIT {limit}" if limit is not None else ""
        
        query = f"""
            SELECT timestamp, machine_group, sensor_tag,
                   sum_value, count_value, min_value, max_value, avg_value, stddev_value,
                   count_invalid, count_missing, count_anomaly,
                   completeness_score, validity_score, anomaly_score, overall_quality_score,
                   aggregation_interval_seconds, expected_count
            FROM aggregated_insights
            WHERE {where_clause}
            ORDER BY timestamp DESC, sensor_tag
            {limit_clause}
        """
        
        # Execute query
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), params)
            
            # Get column names
            columns = result.keys()
            
            # Fetch all rows
            rows = result.fetchall()
            
            # Create DataFrame manually
            df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp column to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for machine group {table}: {str(e)}")

@router.post("/visualization", response_model=VisualizationAnalytics)
def get_visualization_analytics(request: VisualizationRequest):
    """
    Get comprehensive visualization analytics for selected sensor columns
    
    This endpoint provides:
    - Summary statistics for each sensor
    - Correlation matrix between sensors
    - Time series analysis with rolling statistics
    - Histogram and density data
    - Box plot statistics
    - Seasonal decomposition (if applicable)
    - Anomaly detection using Z-score method
    
    Supports:
    - Data source selection (raw/aggregated/auto-detect)
    - Date range filtering
    """
    try:
        # Determine data source type
        if request.data_source == DataSourceType.AUTO:
            detected_source = _detect_data_source_type(request.table)
        else:
            detected_source = request.data_source
        
        # Fetch data from database with date filtering
        # Filter by requested sensor columns if provided
        raw_df = _fetch_table_dataframe(
            request.table, 
            request.limit, 
            request.date_from, 
            request.date_to,
            sensor_tags=request.columns if request.columns else None
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for machine group {request.table}")
        
        # Process aggregated data from TimescaleDB
        # Data comes as rows (timestamp, sensor_tag, avg_value, etc.)
        # Need to pivot to have sensors as columns for the analyzer
        if 'avg_value' in raw_df.columns:
            # Pivot to have sensors as columns
            df_pivot = raw_df.pivot(index='timestamp', columns='sensor_tag', values='avg_value')
            df_pivot = df_pivot.reset_index()
            df_pivot['timestamp'] = pd.to_datetime(df_pivot['timestamp'])
        else:
            raise HTTPException(status_code=500, detail="Missing avg_value column in aggregated data")
        
        # Get available sensors
        available_sensors = [col for col in df_pivot.columns if col != 'timestamp']
        
        # Validate requested columns exist
        if request.columns:
            missing_columns = [col for col in request.columns if col not in available_sensors]
            if missing_columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Sensors not found: {missing_columns}. Available sensors: {available_sensors}"
                )
            # Filter to only requested columns
            columns_to_use = ['timestamp'] + request.columns
            df = df_pivot[columns_to_use]
        else:
            # Use all available sensors
            df = df_pivot
        
        # Create analyzer and perform analysis
        analyzer = create_visualization_analyzer()
        
        # Get sensor columns (exclude timestamp)
        sensor_columns = [col for col in df.columns if col != 'timestamp']
        if request.columns:
            # Use requested columns if provided
            columns_to_analyze = [col for col in request.columns if col in sensor_columns]
        else:
            # Use all available sensor columns
            columns_to_analyze = sensor_columns
        
        if not columns_to_analyze:
            raise HTTPException(status_code=400, detail="No valid sensor columns found for analysis")
        
        analytics = analyzer.analyze_sensors(
            df=df,
            selected_columns=columns_to_analyze,
            table_name=request.table,
            rolling_window=request.rolling_window,
            anomaly_threshold=request.anomaly_threshold,
            seasonal_period=request.seasonal_period
        )
        
        # Add processing info
        analytics.processing_info.update({
            "data_source_type": detected_source.value,
            "date_from": request.date_from.isoformat() if request.date_from else None,
            "date_to": request.date_to.isoformat() if request.date_to else None,
            "total_rows_fetched": len(df)
        })
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing visualization analytics: {str(e)}")

@router.get("/visualization", response_model=VisualizationAnalytics)
def get_visualization_analytics_get(
    table: str = Query(..., description="Table name to analyze"),
    columns: List[str] = Query(..., description="List of column names to analyze"),
    limit: int = Query(5000, description="Maximum number of rows to fetch"),
    rolling_window: int = Query(24, description="Rolling window size for time series analysis"),
    anomaly_threshold: float = Query(2.0, description="Z-score threshold for anomaly detection"),
    seasonal_period: int = Query(24, description="Period for seasonal decomposition"),
    data_source: DataSourceType = Query(DataSourceType.AUTO, description="Data source type: raw, aggregated, or auto-detect"),
    date_from: Optional[date] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="End date for filtering (YYYY-MM-DD)")
):
    """
    GET version of visualization analytics endpoint with data source and date filtering support
    """
    request = VisualizationRequest(
        table=table,
        columns=columns,
        limit=limit,
        rolling_window=rolling_window,
        anomaly_threshold=anomaly_threshold,
        seasonal_period=seasonal_period,
        data_source=data_source,
        date_from=date_from,
        date_to=date_to
    )
    
    return get_visualization_analytics(request)

@router.get("/tables-info")
def get_tables_with_data_source_info():
    """
    Get list of available machine groups with their data source types and date ranges
    """
    try:
        engine = get_engine()
        
        # Query machine groups from aggregated_insights
        query = """
            SELECT 
                machine_group,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(DISTINCT sensor_tag) as sensor_count
            FROM aggregated_insights
            GROUP BY machine_group
            ORDER BY machine_group
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query))
            rows = result.fetchall()
        
        tables_info = []
        for row in rows:
            machine_group, min_date, max_date, sensor_count = row
            
            date_range = None
            if min_date and max_date:
                min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)
                max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
                date_range = {
                    "min_date": min_date_str,
                    "max_date": max_date_str
                }
            
            tables_info.append({
                "table_name": machine_group,
                "machine_group": machine_group,
                "data_source_type": "aggregated",
                "has_timestamp": True,
                "timestamp_column": "timestamp",
                "date_range": date_range,
                "sensor_count": sensor_count,
                "sample_columns": []  # Will be populated on demand
            })
        
        return {
            "tables": tables_info,
            "total_tables": len(tables_info)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tables info: {str(e)}")

@router.post("/invalid-values", response_model=InvalidValuesAnalytics)
def get_invalid_values_analytics(request: InvalidValuesRequest):
    """
    Get comprehensive invalid values/alarms analytics for sensor data
    
    This endpoint provides:
    - Overall statistics (total alarms, avg per sensor, max sensor)
    - Per-sensor statistics (total alarms, percentage, time series)
    - Invalid reading points with timestamps and alarm counts
    
    Designed for aggregated data tables with alarm columns
    """
    try:
        # Fetch data from database with date filtering
        raw_df = _fetch_table_dataframe(
            request.table,
            request.limit,
            request.date_from,
            request.date_to,
            sensor_tags=request.columns if request.columns else None
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for machine group {request.table}")
        
        # Check for required columns in aggregated_insights
        required_cols = ['count_invalid', 'count_value', 'sensor_tag']
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}. This endpoint requires aggregated_insights data."
            )
        
        # Get expected readings per hour from the data
        expected_readings_per_hour = 360  # Default: 10-second frequency = 360 readings/hour
        if 'expected_count' in raw_df.columns:
            expected_mode = raw_df['expected_count'].mode()
            if len(expected_mode) > 0 and expected_mode.iloc[0] > 0:
                inferred = int(expected_mode.iloc[0])
                if 300 <= inferred <= 400:  # Reasonable range around 360
                    expected_readings_per_hour = inferred
        
        # Create analyzer and perform analysis
        analyzer = create_invalid_values_analyzer()
        
        analytics = analyzer.analyze_invalid_values(
            df=raw_df,
            selected_columns=request.columns,
            table_name=request.table,
            threshold=request.threshold,
            expected_readings_per_hour=expected_readings_per_hour
        )
        
        # Add processing info
        analytics.processing_info.update({
            "date_from": request.date_from.isoformat() if request.date_from else None,
            "date_to": request.date_to.isoformat() if request.date_to else None,
            "total_rows_fetched": len(raw_df)
        })
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invalid values analytics: {str(e)}")

@router.get("/invalid-values", response_model=InvalidValuesAnalytics)
def get_invalid_values_analytics_get(
    table: str = Query(..., description="Table name to analyze"),
    columns: Optional[List[str]] = Query(None, description="List of sensor names to analyze"),
    threshold: int = Query(1, description="Minimum alarm count to consider invalid"),
    limit: int = Query(5000, description="Maximum number of rows to fetch"),
    date_from: Optional[date] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="End date for filtering (YYYY-MM-DD)")
):
    """
    GET version of invalid values analytics endpoint with date filtering support
    """
    request = InvalidValuesRequest(
        table=table,
        columns=columns,
        threshold=threshold,
        limit=limit,
        date_from=date_from,
        date_to=date_to
    )
    
    return get_invalid_values_analytics(request)


@router.post("/missing-values", response_model=MissingValuesAnalytics)
def get_missing_values_analytics(request: MissingValuesRequest):
    """
    Analyze missing values in sensor data (POST method)
    """
    try:
        # Validate table exists
        if not request.table:
            raise HTTPException(status_code=400, detail="Machine group name is required")
        
        # Fetch data from aggregated_insights
        raw_df = _fetch_table_dataframe(
            request.table,
            limit=request.limit if request.limit else None,
            date_from=request.date_from,
            date_to=request.date_to,
            sensor_tags=request.columns if request.columns else None
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified criteria")
        
        # Validate required columns exist
        required_cols = ['count_missing', 'count_value', 'completeness_score', 'sensor_tag', 'expected_count']
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}. This endpoint requires aggregated_insights data."
            )
        
        # Get expected readings per hour from the data
        expected_readings_per_hour = 360  # Default
        if 'expected_count' in raw_df.columns:
            expected_mode = raw_df['expected_count'].mode()
            if len(expected_mode) > 0 and expected_mode.iloc[0] > 0:
                inferred = int(expected_mode.iloc[0])
                if 300 <= inferred <= 400:
                    expected_readings_per_hour = inferred
        
        # Create analyzer
        analyzer = create_missing_values_analyzer()
        
        # Analyze missing values using correct frequency
        result = analyzer.analyze_missing_values(
            df=raw_df,
            table_name=request.table,
            selected_columns=request.columns,
            original_freq_sec=10,  # 10 seconds
            expected_readings_per_hour=expected_readings_per_hour  # 360 readings per hour
        )
        
        return MissingValuesAnalytics(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing missing values analytics: {str(e)}")


@router.get("/missing-values", response_model=MissingValuesAnalytics)
def get_missing_values_analytics_get(
    table: str = Query(..., description="Table name to analyze"),
    columns: Optional[List[str]] = Query(None, description="List of sensor names to analyze"),
    limit: Optional[int] = Query(None, description="Maximum number of rows to fetch"),
    date_from: Optional[date] = Query(None, description="Start date for filtering (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="End date for filtering (YYYY-MM-DD)")
):
    """
    GET version of missing values analytics endpoint with date filtering support
    """
    request = MissingValuesRequest(
        table=table,
        columns=columns,
        limit=limit,
        date_from=date_from,
        date_to=date_to
    )
    
    return get_missing_values_analytics(request)


def get_data_quality_analytics(request: DataQualityRequest) -> DataQualityAnalytics:
    """Core function to get data quality analytics"""
    try:
        # Fetch data
        df = _fetch_table_dataframe(
            request.table,
            request.limit,
            request.date_from,
            request.date_to
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified table")
        
        # Load tags data for accuracy checking from database
        tags_df = None
        try:
            engine = get_engine()
            machine_group = request.table  # table parameter represents machine_group
            
            query = """
                SELECT tag, tag_description, machine_group, low_threshold, high_threshold,
                       threshold_type, aggregation_rule, engineering_units, category
                FROM sensor_thresholds
                WHERE machine_group = :machine_group
            """
            
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(query), {"machine_group": machine_group})
                tags_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            print(f"Warning: Could not load tags data from database: {e}")
        
        # Perform analysis
        analyzer = create_data_quality_analyzer()
        analytics = analyzer.analyze_data_quality(
            df,
            request.table,
            tags_df,
            request.completeness_threshold,
            request.correlation_threshold
        )
        
        # Convert to response model
        return DataQualityAnalytics(**analytics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data quality analytics: {str(e)}")


@router.post("/data-quality", response_model=DataQualityAnalytics)
async def post_data_quality_analytics(request: DataQualityRequest):
    """
    POST endpoint for data quality analytics
    """
    return get_data_quality_analytics(request)


@router.get("/data-quality", response_model=DataQualityAnalytics)
async def get_data_quality_analytics_endpoint(
    table: str = Query(..., description="Table name to analyze"),
    limit: Optional[int] = Query(5000, description="Limit number of rows"),
    date_from: Optional[date] = Query(None, description="Start date filter"),
    date_to: Optional[date] = Query(None, description="End date filter"),
    completeness_threshold: float = Query(90.0, description="Completeness threshold percentage"),
    correlation_threshold: float = Query(0.7, description="Correlation threshold for strong correlations")
):
    """
    GET endpoint for data quality analytics
    """
    request = DataQualityRequest(
        table=table,
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        completeness_threshold=completeness_threshold,
        correlation_threshold=correlation_threshold
    )
    
    return get_data_quality_analytics(request)
