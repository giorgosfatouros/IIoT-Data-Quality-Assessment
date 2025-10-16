import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, inspect, Table, MetaData, select
import pandas as pd

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

# Include new routers
app.include_router(visualization_router)
app.include_router(agent_chat_router)
app.include_router(data_import_router)

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
    eng = get_engine()
    all_tables = inspect(eng).get_table_names()
    # Filter to only return _HOURS tables (aggregated data)
    hours_tables = [table for table in all_tables if table.endswith('_HOURS')]
    return hours_tables

@app.get("/tables/{table_name}")
@with_retry(max_retries=3, test_connection=True)
def get_table_info(table_name: str):
    engine = get_engine()
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return {
        "table_name": table_name,
        "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns]
    }

@app.get("/data", response_model=TableData)
def get_data(table: str = Query(...), limit: int = Query(1000)):
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        df = _fetch_table_dataframe(table, limit)
        return TableData(
            columns=df.columns.tolist(),
            rows=df.to_dict(orient="records")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@app.get("/data/preprocessed", response_model=PreprocessedData)
def get_preprocessed_data(table: str = Query(...), limit: int = Query(5000)):
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        df = _fetch_table_dataframe(table, limit)
        
        # Preprocess the data
        processed_df, sensors = _preprocess_sensor_data(df)
        
        # Add timestamp back if it exists
        if 'TIMESTAMP' in df.columns:
            processed_df['timestamp'] = df['TIMESTAMP'].iloc[:len(processed_df)]
        elif 'timestamp' in df.columns:
            processed_df['timestamp'] = df['timestamp'].iloc[:len(processed_df)]
        
        return PreprocessedData(
            sensors=sensors,
            readings=TableData(
                columns=processed_df.columns.tolist(),
                rows=processed_df.to_dict(orient="records")
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

@app.get("/sensors", response_model=List[str])
def get_sensors(table: str = Query(...)):
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        df = _fetch_table_dataframe(table, limit=1)
        _, sensors = _preprocess_sensor_data(df)
        return sensors
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tags", response_model=TagsResponse)
def get_tags(table: Optional[str] = Query(None), limit: int = Query(2000)):
    try:
        # Validate table parameter if provided
        if table and not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        # Load tags CSV from backend/data/tags.csv
        backend_dir = os.path.dirname(__file__)
        tags_path = os.path.join(backend_dir, "data", "tags.csv")
        
        # Check if file exists
        if not os.path.exists(tags_path):
            raise HTTPException(status_code=404, detail=f"Tags file not found at {tags_path}")
        
        tags = pd.read_csv(tags_path, header=0)
        tags.columns = [c.strip().lower().replace(" ", "_") for c in tags.columns]
        
        # Ensure tag column exists and normalize
        if "tag" in tags.columns:
            tags["tag"] = tags["tag"].str.lower()
        else:
            raise HTTPException(status_code=400, detail="Tags CSV must contain a 'tag' column")

        # Filter by table columns if table is specified
        if table:
            try:
                df = _fetch_table_dataframe(table, limit=limit)
                present_cols = set(df.columns)
                
                # Extract tag names from column names (remove prefixes like "COL", "SUM_", "COUNT_", etc.)
                extracted_tags = set()
                for col in present_cols:
                    # Handle columns like "SUM_COL33VI603", "COUNT_COL33VI603", "MIN_COL33VI603", "MAX_COL33VI603"
                    if col.startswith(('SUM_COL', 'COUNT_COL', 'MIN_COL', 'MAX_COL')):
                        tag = col.split('_', 1)[1]  # Remove "SUM_", "COUNT_", etc.
                        if tag.startswith('COL'):
                            tag = tag[3:]  # Remove "COL" prefix -> "33VI603"
                        extracted_tags.add(tag.lower())  # Convert to lowercase for matching
                    elif col.startswith('COL'):
                        extracted_tags.add(col[3:].lower())  # Remove "COL" prefix and lowercase
                    else:
                        extracted_tags.add(col.lower())  # Convert to lowercase for matching
                
                # Filter tags based on extracted tag names
                tags = tags[tags["tag"].isin(extracted_tags)].reset_index(drop=True)
            except Exception as table_err:
                # If table fetch fails, return all tags with a warning
                print(f"Warning: Could not fetch table {table} for filtering: {table_err}")

        rows = tags.to_dict(orient="records")
        return {"columns": list(tags.columns), "rows": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading tags: {str(e)}")

@app.get("/analytics/aggregation_frequency", response_model=AggregationFrequency)
def get_aggregation_frequency(table: str = Query(...)):
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        df = _fetch_table_dataframe(table, limit=100)
        
        if 'timestamp' not in df.columns:
            return AggregationFrequency(aggregation_frequency_seconds=None)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate time differences
        time_diffs = df['timestamp'].diff().dropna()
        
        if len(time_diffs) == 0:
            return AggregationFrequency(aggregation_frequency_seconds=None)
        
        # Get the most common time difference (mode)
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) > 0:
            freq_seconds = int(mode_diff.iloc[0].total_seconds())
            return AggregationFrequency(aggregation_frequency_seconds=freq_seconds)
        else:
            return AggregationFrequency(aggregation_frequency_seconds=None)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating aggregation frequency: {str(e)}")

@app.get("/analytics/missing", response_model=MissingValuesResponse)
def get_missing_values_analysis(table: str = Query(...), limit: int = Query(5000)):
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table.endswith('_HOURS'):
            raise HTTPException(
                status_code=400, 
                detail="Only _HOURS tables (aggregated data) are supported. Use tables ending with '_HOURS'."
            )
        
        df = _fetch_table_dataframe(table, limit)
        
        # Calculate missing values percentage for each column
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Calculate total missing percentage
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        total_missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
        
        return MissingValuesResponse(
            missing_values=missing_percentages,
            total_missing_percentage=total_missing_percentage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing missing values: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
