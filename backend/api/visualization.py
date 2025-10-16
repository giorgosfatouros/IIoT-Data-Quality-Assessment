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
    """Detect if table contains raw or aggregated data based on table name"""
    table_lower = table.lower()
    if any(suffix in table_lower for suffix in ['_hours', '_daily', '_weekly', '_monthly', '_agg', '_aggregated']):
        return DataSourceType.AGGREGATED
    else:
        return DataSourceType.RAW

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
    date_to: Optional[date] = None
) -> pd.DataFrame:
    """Fetch data from database table and return as DataFrame with optional date filtering"""
    try:
        engine = get_engine()
        
        # Build query with date filtering if provided
        base_query = f"SELECT * FROM {table}"
        where_conditions = []
        
        # First, get a sample to detect timestamp column
        sample_query = f"SELECT * FROM {table} LIMIT 1"
        with engine.connect() as conn:
            from sqlalchemy import text
            sample_result = conn.execute(text(sample_query))
            sample_columns = list(sample_result.keys())
        
        # Find timestamp column
        timestamp_col = None
        timestamp_candidates = ['timestamp', 'TIMESTAMP', 'time', 'TIME', 'date', 'DATE', 'datetime', 'DATETIME']
        for col in timestamp_candidates:
            if col in sample_columns:
                timestamp_col = col
                break
        
        # Add date filtering if timestamp column exists and dates are provided
        if timestamp_col and (date_from or date_to):
            if date_from:
                where_conditions.append(f'"{timestamp_col}" >= \'{date_from}\'')
            if date_to:
                # Add one day to include the entire end date
                end_date = pd.to_datetime(date_to) + pd.Timedelta(days=1)
                where_conditions.append(f'"{timestamp_col}" < \'{end_date.strftime("%Y-%m-%d")}\'')
        
        # Construct final query with optional LIMIT
        limit_clause = f" LIMIT {limit}" if limit is not None else ""
        if where_conditions:
            query = f'SELECT * FROM "{table}" WHERE {" AND ".join(where_conditions)}{limit_clause}'
        else:
            query = f'SELECT * FROM "{table}"{limit_clause}'
        
        # Execute query
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query))
            
            # Get column names
            columns = result.keys()
            
            # Fetch all rows
            rows = result.fetchall()
            
            # Create DataFrame manually
            df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp column to datetime if it exists
        if timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            # Don't set as index here, let the calling function decide
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from table {table}: {str(e)}")

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
        raw_df = _fetch_table_dataframe(
            request.table, 
            request.limit, 
            request.date_from, 
            request.date_to
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found in table {request.table}")
        
        # Process data based on source type
        if detected_source == DataSourceType.AGGREGATED:
            # For aggregated data, preprocess to get sensor means
            from main import _preprocess_sensor_data
            processed_df, sensors = _preprocess_sensor_data(raw_df)
            
            # Add timestamp back if it exists
            timestamp_col = _get_timestamp_column(raw_df)
            if timestamp_col and timestamp_col in raw_df.columns:
                processed_df['timestamp'] = raw_df[timestamp_col].iloc[:len(processed_df)]
            
            # Validate columns exist in preprocessed data
            missing_columns = [col for col in request.columns if col not in processed_df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Columns not found in preprocessed data: {missing_columns}. Available columns: {list(processed_df.columns)}"
                )
            
            df = processed_df
            
        else:  # RAW data
            # For raw data, use columns directly
            timestamp_col = _get_timestamp_column(raw_df)
            if timestamp_col:
                raw_df['timestamp'] = raw_df[timestamp_col]
            
            # Validate columns exist in raw data
            missing_columns = [col for col in request.columns if col not in raw_df.columns]
            if missing_columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Columns not found in raw data: {missing_columns}. Available columns: {list(raw_df.columns)}"
                )
            
            df = raw_df
        
        # Create analyzer and perform analysis
        analyzer = create_visualization_analyzer()
        
        analytics = analyzer.analyze_sensors(
            df=df,
            selected_columns=request.columns,
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
    Get list of available tables with their detected data source types and date ranges
    """
    try:
        from models.database import get_engine
        from sqlalchemy import inspect
        
        engine = get_engine()
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        tables_info = []
        for table in table_names:
            try:
                # Detect data source type
                data_source_type = _detect_data_source_type(table)
                
                # Get sample data to check for timestamp column and date range
                sample_df = _fetch_table_dataframe(table, limit=10)
                timestamp_col = _get_timestamp_column(sample_df)
                
                date_range = None
                if timestamp_col and not sample_df.empty:
                    try:
                        # Get min/max dates using a single query to avoid LeanXcale syntax issues
                        date_range_query = f'SELECT MIN("{timestamp_col}") as min_date, MAX("{timestamp_col}") as max_date FROM "{table}"'
                        
                        with engine.connect() as conn:
                            from sqlalchemy import text
                            result = conn.execute(text(date_range_query)).fetchone()
                            
                            if result and result[0] and result[1]:
                                min_date = result[0]
                                max_date = result[1]
                                
                                # Convert to string format
                                min_date_str = min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date)
                                max_date_str = max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
                                
                                date_range = {
                                    "min_date": min_date_str,
                                    "max_date": max_date_str
                                }
                    except Exception as date_error:
                        # If date range query fails, just skip it
                        print(f"Warning: Could not get date range for table {table}: {date_error}")
                
                tables_info.append({
                    "table_name": table,
                    "data_source_type": data_source_type.value,
                    "has_timestamp": timestamp_col is not None,
                    "timestamp_column": timestamp_col,
                    "date_range": date_range,
                    "sample_columns": list(sample_df.columns) if not sample_df.empty else []
                })
                
            except Exception as table_error:
                # If we can't analyze a table, still include it with basic info
                tables_info.append({
                    "table_name": table,
                    "data_source_type": _detect_data_source_type(table).value,
                    "has_timestamp": None,
                    "timestamp_column": None,
                    "date_range": None,
                    "sample_columns": [],
                    "error": str(table_error)
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
            request.date_to
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found in table {request.table}")
        
        # Check for required COUNT columns (for calculating invalid values)
        count_columns = [col for col in raw_df.columns if col.startswith('COUNT_COL') and '_ISVALID' not in col]
        count_isvalid_columns = [col for col in raw_df.columns if 'COUNT_' in col and '_ISVALID' in col]
        
        if not count_columns or not count_isvalid_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Table {request.table} does not have the required COUNT and COUNT_ISVALID columns. This endpoint requires aggregated data."
            )
        
        # Expected readings per hour for 10-second frequency data
        # Original data is collected every 10 SECONDS, so 360 readings per hour (3600/10=360)
        # _HOURS tables aggregate these 360 readings per hour
        try:
            # Try to infer from timestamps or COUNT columns
            timestamp_col = _get_timestamp_column(raw_df)
            expected_readings_per_hour = 360  # Default: 10-second frequency = 360 readings/hour
            
            # Try to infer from actual COUNT values (should be around 360 if data is complete)
            count_cols = [col for col in raw_df.columns if col.startswith('COUNT_COL') and '_ISVALID' not in col]
            if count_cols:
                # Get the mode (most common) COUNT value
                count_mode = raw_df[count_cols[0]].mode()
                if len(count_mode) > 0 and count_mode.iloc[0] > 0:
                    inferred = int(count_mode.iloc[0])
                    # If mode is around 360, use it; otherwise stick with expected 360
                    if 300 <= inferred <= 400:  # Reasonable range around 360
                        expected_readings_per_hour = inferred
                    
            if timestamp_col and timestamp_col in raw_df.columns:
                raw_df[timestamp_col] = pd.to_datetime(raw_df[timestamp_col])
                time_diffs = raw_df[timestamp_col].diff().dropna()
                
                if len(time_diffs) > 0:
                    mode_diff = time_diffs.mode()
                    if len(mode_diff) > 0:
                        agg_freq_seconds = int(mode_diff.iloc[0].total_seconds())
                        # If data is aggregated hourly (3600 seconds)
                        # and original frequency is 10 seconds, we expect COUNT=360
                        if agg_freq_seconds == 3600:  # Hourly aggregation
                            # Keep the inferred value from COUNT columns
                            pass
        except Exception as e:
            print(f"Could not infer aggregation frequency: {e}")
            expected_readings_per_hour = 360
            
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
            raise HTTPException(status_code=400, detail="Table name is required")
        
        # Fetch data
        raw_df = _fetch_table_dataframe(
            request.table,
            limit=request.limit if request.limit else None,
            date_from=request.date_from,
            date_to=request.date_to
        )
        
        if raw_df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified criteria")
        
        # Validate it's aggregated data with COUNT columns
        count_cols = [col for col in raw_df.columns if col.startswith('COUNT_COL') and '_ISVALID' not in col]
        if not count_cols:
            raise HTTPException(
                status_code=400, 
                detail="Table must contain COUNT_COL* columns (aggregated data). Use *_HOURS tables."
            )
        
        # Expected readings per hour for 10-second frequency data
        # Data is collected every 10 seconds, so 360 readings per hour (3600/10=360)
        expected_readings_per_hour = 360
        
        # Optionally infer from actual data (should be close to 360 if data is complete)
        try:
            if count_cols:
                count_mode = raw_df[count_cols[0]].mode()
                if len(count_mode) > 0 and count_mode.iloc[0] > 0:
                    # Use the mode value if it's reasonable
                    inferred = int(count_mode.iloc[0])
                    if 300 <= inferred <= 400:  # Reasonable range around 360
                        expected_readings_per_hour = inferred
        except Exception as e:
            print(f"Could not infer readings per hour: {e}")
        
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
        
        # Load tags data for accuracy checking
        tags_df = None
        try:
            # Try to load tags data from tags.csv
            tags_path = os.path.join(os.path.dirname(__file__), '../data/tags.csv')
            if os.path.exists(tags_path):
                tags_df = pd.read_csv(tags_path)
        except Exception as e:
            print(f"Warning: Could not load tags data: {e}")
        
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
