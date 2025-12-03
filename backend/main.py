import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, inspect, Table, MetaData, select
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Import new modules
from api.visualization import router as visualization_router
from api.agent_chat import router as agent_chat_router
from api.data_import import router as data_import_router
from models.database import get_engine
from core.connection_manager import with_retry, execute_with_retry

# Keep existing Pydantic models for backward compatibility
class TableData(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    version: str

class PreprocessedData(BaseModel):
    sensors: List[str]
    readings: TableData

class TagsResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

class AggregationFrequency(BaseModel):
    aggregation_frequency_seconds: Optional[int]

class MissingValuesResponse(BaseModel):
    missing_values: Dict[str, float]
    total_missing_percentage: float

# Create FastAPI app
app = FastAPI(title="IIoT Data Quality API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add validation error handler to debug 422 errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging"""
    logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )

# Include new routers
app.include_router(visualization_router)
app.include_router(agent_chat_router)
app.include_router(data_import_router)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker and load balancers"""
    try:
        # Test database connection
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        timestamp=datetime.now().isoformat(),
        database=db_status,
        version="0.2.0"
    )

# Helper functions
@with_retry(max_retries=5, test_connection=True)
def _fetch_table_dataframe(table: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch data from database table and return as DataFrame with retry logic"""
    engine = get_engine()
    query = f"SELECT * FROM {table} LIMIT {limit}"
    
    # Use SQLAlchemy's execute method instead of pandas read_sql for LeanXcale compatibility
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text(query))
        
        # Get column names
        columns = result.keys()
        
        # Fetch all rows
        rows = result.fetchall()
        
        # Create DataFrame manually
        df = pd.DataFrame(rows, columns=columns)
        
        # Explicitly close the result to prevent connection issues
        result.close()
    
    return df

def _preprocess_sensor_data(df: pd.DataFrame) -> tuple:
    """Preprocess sensor data to extract sensors and calculate means"""
    # Identify unique sensors - look for SUM_ columns (uppercase)
    # Format: SUM_COL33VI603 -> extract COL33VI603
    sensors = list(set(col.split('_', 1)[1] 
                      for col in df.columns if col.startswith('SUM_')))
    
    # Calculate mean value per sensor
    mean_values_per_sensor = {}
    for sensor in sensors:
        sum_col = f'SUM_{sensor}'
        count_col = f'COUNT_{sensor}'
        mean_col = f'mean_{sensor}'
        
        if sum_col in df.columns and count_col in df.columns:
            # Avoid division by zero
            count_series = df[count_col].replace(0, pd.NA)
            mean_values_per_sensor[mean_col] = df[sum_col] / count_series
    
    # Convert to DataFrame
    mean_df = pd.DataFrame(mean_values_per_sensor)
    
    # Extract sensor IDs (remove 'COL' prefix if present)
    clean_sensors = [s.replace('COL', '') for s in sensors]
    
    # Update column names to use clean sensor names
    if len(clean_sensors) == len(mean_df.columns):
        mean_df.columns = clean_sensors
    
    return mean_df, clean_sensors

# Existing endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tables", response_model=List[str])
@with_retry(max_retries=3, test_connection=True)
def list_tables():
    """List available machine groups from aggregated_insights table"""
    engine = get_engine()
    query = "SELECT DISTINCT machine_group FROM aggregated_insights ORDER BY machine_group"
    
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text(query))
        machine_groups = [row[0] for row in result.fetchall()]
    
    return machine_groups if machine_groups else []

@app.get("/tables/{machine_group}")
@with_retry(max_retries=3, test_connection=True)
def get_table_info(machine_group: str):
    """Get information about sensors for a machine group"""
    engine = get_engine()
    query = """
        SELECT DISTINCT sensor_tag
        FROM aggregated_insights
        WHERE machine_group = :machine_group
        ORDER BY sensor_tag
    """
    
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text(query), {"machine_group": machine_group})
        sensors = [row[0] for row in result.fetchall()]
    
    return {
        "table_name": machine_group,
        "machine_group": machine_group,
        "sensors": sensors,
        "sensor_count": len(sensors)
    }

@app.get("/data", response_model=TableData)
def get_data(table: str = Query(...), limit: int = Query(1000)):
    """Get aggregated insights data for a machine group (table parameter is now machine_group)"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        query = """
            SELECT timestamp, machine_group, sensor_tag, 
                   sum_value, count_value, min_value, max_value, avg_value, stddev_value,
                   count_invalid, count_missing, count_anomaly,
                   completeness_score, validity_score, anomaly_score, overall_quality_score
            FROM aggregated_insights
            WHERE machine_group = :machine_group
            ORDER BY timestamp DESC, sensor_tag
            LIMIT :limit
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), {"machine_group": machine_group, "limit": limit})
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        return TableData(
            columns=df.columns.tolist(),
            rows=df.to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.get("/data/preprocessed", response_model=PreprocessedData)
def get_preprocessed_data(table: str = Query(...), limit: int = Query(5000)):
    """Get preprocessed sensor data (pivoted by sensor) for a machine group"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        # Fetch aggregated insights
        query = """
            SELECT timestamp, sensor_tag, avg_value as mean_value
            FROM aggregated_insights
            WHERE machine_group = :machine_group
            ORDER BY timestamp DESC, sensor_tag
            LIMIT :limit
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), {"machine_group": machine_group, "limit": limit})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            return PreprocessedData(
                sensors=[],
                readings=TableData(columns=[], rows=[])
            )
        
        # Pivot to have sensors as columns
        df_pivot = df.pivot(index='timestamp', columns='sensor_tag', values='mean_value')
        df_pivot = df_pivot.reset_index()
        
        # Get sensor list
        sensors = [col for col in df_pivot.columns if col != 'timestamp']
        
        return PreprocessedData(
            sensors=sensors,
            readings=TableData(
                columns=df_pivot.columns.tolist(),
                rows=df_pivot.to_dict(orient="records")
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

@app.get("/sensors", response_model=List[str])
def get_sensors(table: str = Query(...)):
    """Get list of sensors for a machine group"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        query = """
            SELECT DISTINCT sensor_tag
            FROM aggregated_insights
            WHERE machine_group = :machine_group
            ORDER BY sensor_tag
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), {"machine_group": machine_group})
            sensors = [row[0] for row in result.fetchall()]
        
        return sensors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tags", response_model=TagsResponse)
def get_tags(table: Optional[str] = Query(None), limit: int = Query(2000)):
    """Get sensor tags/thresholds from sensor_thresholds table"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        # Build query
        if machine_group:
            # Filter by machine group
            query = """
                SELECT tag, tag_description, machine_group, low_threshold, high_threshold,
                       threshold_type, aggregation_rule, engineering_units, category
                FROM sensor_thresholds
                WHERE machine_group = :machine_group
                ORDER BY tag
                LIMIT :limit
            """
            params = {"machine_group": machine_group, "limit": limit}
        else:
            # Get all tags
            query = """
                SELECT tag, tag_description, machine_group, low_threshold, high_threshold,
                       threshold_type, aggregation_rule, engineering_units, category
                FROM sensor_thresholds
                ORDER BY machine_group, tag
                LIMIT :limit
            """
            params = {"limit": limit}
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), params)
            columns = result.keys()
            rows = result.fetchall()
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(rows, columns=columns)
        
        # Convert to response format
        rows_dict = df.to_dict(orient="records")
        return {"columns": list(df.columns), "rows": rows_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading tags: {str(e)}")

@app.get("/analytics/aggregation_frequency", response_model=AggregationFrequency)
def get_aggregation_frequency(table: str = Query(...)):
    """Get aggregation frequency for a machine group"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        query = """
            SELECT DISTINCT aggregation_interval_seconds
            FROM aggregated_insights
            WHERE machine_group = :machine_group
            LIMIT 1
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), {"machine_group": machine_group})
            row = result.fetchone()
            
            if row and row[0]:
                return AggregationFrequency(aggregation_frequency_seconds=int(row[0]))
            else:
                # Default to 3600 seconds (1 hour) if not found
                return AggregationFrequency(aggregation_frequency_seconds=3600)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating aggregation frequency: {str(e)}")

@app.get("/analytics/missing", response_model=MissingValuesResponse)
def get_missing_values_analysis(table: str = Query(...), limit: int = Query(5000)):
    """Get missing values analysis for a machine group from aggregated_insights"""
    try:
        engine = get_engine()
        machine_group = table  # table parameter now represents machine_group
        
        # Query aggregated insights for missing values data
        query = """
            SELECT sensor_tag, 
                   AVG(count_missing) as avg_missing,
                   AVG(count_value) as avg_count,
                   AVG(expected_count) as avg_expected,
                   AVG(completeness_score) as avg_completeness
            FROM aggregated_insights
            WHERE machine_group = :machine_group
            GROUP BY sensor_tag
            ORDER BY sensor_tag
            LIMIT :limit
        """
        
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(query), {"machine_group": machine_group, "limit": limit})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        if df.empty:
            return MissingValuesResponse(
                missing_values={},
                total_missing_percentage=0.0
            )
        
        # Calculate missing percentage per sensor
        missing_percentages = {}
        for _, row in df.iterrows():
            sensor = row['sensor_tag']
            avg_expected = row['avg_expected'] or 360  # Default to 360 if null
            avg_missing = row['avg_missing'] or 0
            missing_pct = (avg_missing / avg_expected * 100) if avg_expected > 0 else 0.0
            missing_percentages[sensor] = float(missing_pct)
        
        # Calculate total missing percentage (average of all sensors)
        total_missing_percentage = float(df['avg_completeness'].mean()) if 'avg_completeness' in df.columns else 0.0
        # Convert completeness to missing percentage
        total_missing_percentage = 100.0 - total_missing_percentage if total_missing_percentage > 0 else 0.0
        
        return MissingValuesResponse(
            missing_values=missing_percentages,
            total_missing_percentage=total_missing_percentage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing missing values: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
