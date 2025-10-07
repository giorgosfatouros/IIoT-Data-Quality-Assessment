import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, inspect, Table, MetaData, select
import pandas as pd


DB_USER = os.getenv("DB_USER", "app")
DB_PASS = os.getenv("DB_PASS", "app")
DB_IP = os.getenv("DB_IP", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "1529")
DB_NAME = os.getenv("DB_NAME", "MOH")


def get_engine():
    # Keep the same URL style as the Streamlit PoC
    url = (
        f"leanxcale://{DB_USER}:{DB_PASS}@{DB_IP}:{DB_PORT}/{DB_NAME}"
        f"?autocommit=False&parallel=True&txn_mode=NO_CONFLICTS_NO_LOGGING"
    )
    return create_engine(url)


app = FastAPI(title="IIoT Data Quality API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tables", response_model=List[str])
def list_tables():
    try:
        eng = get_engine()
        return inspect(eng).get_table_names()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QueryParams(BaseModel):
    limit: int = 100


@app.get("/tables/{table_name}")
def read_table_sample(table_name: str, limit: int = 100) -> Dict[str, Any]:
    try:
        eng = get_engine()
        metadata = MetaData(bind=eng)
        table = Table(table_name, metadata, autoload=True)
        stmt = select([table]).limit(limit)
        with eng.connect() as conn:
            result = conn.execute(stmt)
            rows = [dict(r) for r in result.fetchall()]
        return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =========================
# Schemas
# =========================

class TableRowsResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class PreprocessedResponse(BaseModel):
    sensors: List[str]
    readings: TableRowsResponse


class TagsResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class AggregationFrequencyResponse(BaseModel):
    aggregation_frequency_seconds: Optional[int]


class MissingValuesResponse(BaseModel):
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    missing_intervals: Dict[str, List[List[str]]]


# =========================
# Helpers
# =========================

def _fetch_table_dataframe(table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    eng = get_engine()
    metadata = MetaData(bind=eng)
    table = Table(table_name, metadata, autoload=True)
    stmt = select([table])
    if limit:
        stmt = stmt.limit(limit)
    with eng.connect() as conn:
        result = conn.execute(stmt)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    # Normalize column names to lowercase to align with Streamlit PoC expectations
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _infer_aggregation_frequency_seconds(df: pd.DataFrame) -> Optional[int]:
    ts_col = None
    for cand in ["timestamp", "time", "ts", "reading_time"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        return None
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    ts = ts.dropna().sort_values()
    if ts.empty or len(ts) < 2:
        return None
    diffs = ts.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return None
    # Use mode as in the PoC
    try:
        inferred = int(diffs.mode().iloc[0])
        return inferred if inferred > 0 else None
    except Exception:
        return None


def _preprocess_sensor_data(df: pd.DataFrame) -> (pd.DataFrame, List[str]):
    # Mirrors utils.preprocess_sensor_data logic
    sensors = list(set(col.split("_")[1] for col in df.columns if col.startswith("sum_")))
    mean_values_per_sensor: Dict[str, Any] = {}
    alarms_per_sensor: Dict[str, Any] = {}
    for sensor in sensors:
        sum_col = f"sum_{sensor}"
        count_col = f"count_{sensor}"
        mean_col = f"mean_{sensor}"
        alarm_col = f"count_{sensor}_isvalid"
        alarms_col = f"{sensor}_alarms"
        if sum_col in df.columns and count_col in df.columns:
            # Avoid division by zero
            denom = df[count_col].replace(0, pd.NA)
            mean_values_per_sensor[mean_col] = (df[sum_col] / denom).astype(float)
        if alarm_col in df.columns:
            alarms_per_sensor[alarms_col] = df[alarm_col]
    mean_df = pd.DataFrame(mean_values_per_sensor)
    alarms_df = pd.DataFrame(alarms_per_sensor)
    alarms_df.columns = [col.replace("col", "") for col in alarms_df.columns]
    sensors_norm = [s.replace("col", "") for s in sensors]
    mean_df.columns = sensors_norm
    result_df = pd.concat([mean_df, alarms_df], axis=1)
    return result_df, sensors_norm


def _identify_intervals(timestamps: pd.Series, freq_sec: int) -> List[List[str]]:
    if timestamps.empty:
        return []
    ts_sorted = timestamps.sort_values()
    intervals: List[List[str]] = []
    start = ts_sorted.iloc[0]
    end = ts_sorted.iloc[0]
    for t in ts_sorted.iloc[1:]:
        if (t - end).total_seconds() <= freq_sec:
            end = t
        else:
            intervals.append([start.isoformat(), end.isoformat()])
            start = t
            end = t
    intervals.append([start.isoformat(), end.isoformat()])
    return intervals


# =========================
# New Endpoints
# =========================

@app.get("/data", response_model=TableRowsResponse)
def get_table_data(table: str = Query(..., description="Table name"), limit: int = Query(1000)):
    try:
        df = _fetch_table_dataframe(table, limit=limit)
        rows = df.to_dict(orient="records")
        return {"columns": list(df.columns), "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/preprocessed", response_model=PreprocessedResponse)
def get_preprocessed_data(table: str = Query(...), limit: int = Query(5000)):
    try:
        raw = _fetch_table_dataframe(table, limit=limit)
        readings, sensors = _preprocess_sensor_data(raw)
        rows = readings.to_dict(orient="records")
        return {"sensors": sensors, "readings": {"columns": list(readings.columns), "rows": rows}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tags", response_model=TagsResponse)
def get_tags(table: Optional[str] = Query(None), limit: int = Query(2000)):
    try:
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
                
                # Extract tag names from column names (remove prefixes like "col", "sum_", "count_", etc.)
                extracted_tags = set()
                for col in present_cols:
                    # Handle columns like "sum_col33vi603", "count_col33vi603", "col33vi603"
                    if col.startswith(('sum_col', 'count_col', 'min_col', 'max_col')):
                        tag = col.split('_', 1)[1]  # Remove "sum_", "count_", etc.
                        if tag.startswith('col'):
                            tag = tag[3:]  # Remove "col" prefix
                        extracted_tags.add(tag)
                    elif col.startswith('col'):
                        extracted_tags.add(col[3:])  # Remove "col" prefix
                    else:
                        extracted_tags.add(col)
                
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


@app.get("/analytics/aggregation_frequency", response_model=AggregationFrequencyResponse)
def get_aggregation_frequency(table: str = Query(...), limit: int = Query(10000)):
    try:
        df = _fetch_table_dataframe(table, limit=limit)
        agg = _infer_aggregation_frequency_seconds(df)
        return {"aggregation_frequency_seconds": agg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/missing", response_model=MissingValuesResponse)
def get_missing_values(table: str = Query(...), orig_freq: int = Query(10, description="Original data frequency in seconds")):
    try:
        df = _fetch_table_dataframe(table, limit=None)
        # Identify timestamp column
        ts_col = None
        for cand in ["timestamp", "time", "ts", "reading_time"]:
            if cand in df.columns:
                ts_col = cand
                break
        if ts_col is None:
            raise HTTPException(status_code=400, detail="No timestamp column found in table")
        ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
        if ts.empty:
            raise HTTPException(status_code=400, detail="No valid timestamps in table")
        total_duration_sec = int((ts.iloc[-1] - ts.iloc[0]).total_seconds())
        expected_readings = max(total_duration_sec // int(orig_freq), 1)

        count_columns = [c for c in df.columns if c.startswith("count_") and not c.endswith("_isvalid")]
        missing_counts: Dict[str, int] = {}
        missing_percentages: Dict[str, float] = {}
        missing_intervals: Dict[str, List[List[str]]] = {}

        # For intervals, we consider missing timestamps where count < expected per bucket (approximate)
        inferred = _infer_aggregation_frequency_seconds(df) or orig_freq
        ts_series = pd.to_datetime(df[ts_col], errors="coerce")

        for col in count_columns:
            total_seen = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            missing = int(max(expected_readings - total_seen, 0))
            missing_counts[col] = missing
            missing_percentages[col] = (missing / expected_readings) * 100.0

            # Find timestamps where this bucket is short of the expected aggregation (heuristic)
            short_ts = ts_series[pd.to_numeric(df[col], errors="coerce").fillna(0) < max(inferred // orig_freq, 1)]
            missing_intervals[col] = _identify_intervals(short_ts.dropna(), max(inferred, orig_freq))

        return {
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "missing_intervals": missing_intervals,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

