"""
DQA Agent Chat API - OpenAI Agents SDK integration for data quality questions
"""
import os
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import inspect, text
import json

from agents import Agent, Runner, function_tool
from models.database import get_engine

router = APIRouter(prefix="/agent", tags=["agent"])

# Request/Response models
class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    table_name: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    agent_name: str
    conversation_history: List[ChatMessage]


# Tool functions that the agent can use
@function_tool
def list_available_tables() -> str:
    """
    List all available aggregated data tables in the database.
    Only shows _HOURS tables which contain aggregated sensor data.
    
    Returns:
        A formatted string listing all _HOURS table names
    """
    try:
        engine = get_engine()
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        # Filter to only return _HOURS tables (aggregated data)
        hours_tables = [table for table in all_tables if table.endswith('_HOURS')]
        if not hours_tables:
            return "No aggregated data tables (_HOURS) found in the database."
        return "Available aggregated data tables:\n" + "\n".join(f"- {table}" for table in hours_tables)
    except Exception as e:
        return f"Error listing tables: {str(e)}"


@function_tool
def get_table_schema(table_name: str) -> str:
    """
    Get the schema (column names and types) for a specific aggregated data table.
    Only works with _HOURS tables which contain aggregated sensor data.
    
    Args:
        table_name: Name of the _HOURS table to inspect
        
    Returns:
        A formatted string showing column names and their data types
    """
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table_name.endswith('_HOURS'):
            return f"Error: Only _HOURS tables (aggregated data) are supported. Table '{table_name}' is not an aggregated data table."
        
        engine = get_engine()
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        
        if not columns:
            return f"No columns found for table '{table_name}'."
        
        schema_info = f"Schema for aggregated data table '{table_name}':\n"
        for col in columns:
            schema_info += f"- {col['name']}: {col['type']}\n"
        
        return schema_info
    except Exception as e:
        return f"Error getting schema for table '{table_name}': {str(e)}"


@function_tool
def query_sensor_data(table_name: str, limit: int = 100) -> str:
    """
    Query aggregated sensor data from a specific _HOURS table and get summary statistics.
    This is useful for understanding data quality, completeness, and patterns in aggregated data.
    
    Args:
        table_name: Name of the _HOURS table to query (aggregated data only)
        limit: Maximum number of rows to fetch (default 100)
        
    Returns:
        Summary statistics and data quality information for aggregated sensor data
    """
    try:
        # Validate that only _HOURS tables (aggregated data) are used
        if not table_name.endswith('_HOURS'):
            return f"Error: Only _HOURS tables (aggregated data) are supported. Table '{table_name}' is not an aggregated data table."
        
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        if df.empty:
            return f"Table '{table_name}' is empty or has no data."
        
        # Generate summary
        summary = f"Data summary for table '{table_name}' (sampled {len(df)} rows):\n\n"
        
        # Basic info
        summary += f"Total columns: {len(df.columns)}\n"
        summary += f"Total rows sampled: {len(df)}\n\n"
        
        # Data quality metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        summary += f"Data Quality Metrics:\n"
        summary += f"- Total cells: {total_cells}\n"
        summary += f"- Missing cells: {missing_cells} ({missing_percentage:.2f}%)\n"
        summary += f"- Complete cells: {total_cells - missing_cells} ({100 - missing_percentage:.2f}%)\n\n"
        
        # Per-column missing values
        missing_per_col = df.isnull().sum()
        if missing_per_col.sum() > 0:
            summary += "Missing values by column:\n"
            for col, missing_count in missing_per_col[missing_per_col > 0].items():
                pct = (missing_count / len(df) * 100)
                summary += f"- {col}: {missing_count} ({pct:.2f}%)\n"
            summary += "\n"
        
        # Identify sensor columns (SUM_, COUNT_, MIN_, MAX_)
        sensor_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['SUM_', 'COUNT_', 'MIN_', 'MAX_'])]
        if sensor_cols:
            summary += f"Sensor columns found: {len(sensor_cols)}\n"
            unique_sensors = set()
            for col in sensor_cols:
                if '_' in col:
                    sensor_name = '_'.join(col.split('_')[1:])
                    # Filter out _ISVALID tracking columns (not actual sensors)
                    if not sensor_name.endswith('_ISVALID'):
                        unique_sensors.add(sensor_name)
            summary += f"Unique sensors: {len(unique_sensors)}\n"
            if len(unique_sensors) <= 10:
                summary += f"Sensor names: {', '.join(list(unique_sensors)[:10])}\n"
        
        return summary
    except Exception as e:
        return f"Error querying table '{table_name}': {str(e)}"


@function_tool
def analyze_sensor_statistics(table_name: str, sensor_name: str, limit: int = 1000) -> str:
    """
    Get detailed statistics for a specific sensor including mean, min, max, and data quality.
    
    Args:
        table_name: Name of the table containing sensor data
        sensor_name: Name of the sensor (e.g., '33VI603' or 'COL33VI603')
        limit: Number of rows to analyze (default 1000)
        
    Returns:
        Detailed statistics and quality metrics for the sensor
    """
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        # Find columns for this sensor
        sensor_cols = [col for col in df.columns if sensor_name.upper() in col.upper()]
        
        if not sensor_cols:
            return f"No columns found for sensor '{sensor_name}' in table '{table_name}'."
        
        stats = f"Statistics for sensor '{sensor_name}' in table '{table_name}':\n\n"
        
        # Check for aggregated columns
        sum_col = next((col for col in sensor_cols if col.startswith('SUM_')), None)
        count_col = next((col for col in sensor_cols if col.startswith('COUNT_')), None)
        min_col = next((col for col in sensor_cols if col.startswith('MIN_')), None)
        max_col = next((col for col in sensor_cols if col.startswith('MAX_')), None)
        
        if sum_col and count_col:
            # Calculate mean values
            count_series = df[count_col].replace(0, pd.NA)
            mean_values = df[sum_col] / count_series
            
            stats += f"Aggregated Statistics (from {len(df)} rows):\n"
            stats += f"- Mean of means: {mean_values.mean():.2f}\n"
            stats += f"- Std deviation: {mean_values.std():.2f}\n"
            stats += f"- Min mean: {mean_values.min():.2f}\n"
            stats += f"- Max mean: {mean_values.max():.2f}\n"
            stats += f"- Missing values: {mean_values.isnull().sum()} ({mean_values.isnull().sum() / len(df) * 100:.2f}%)\n\n"
        
        if min_col:
            stats += f"Minimum values:\n"
            stats += f"- Overall min: {df[min_col].min():.2f}\n"
            stats += f"- Missing: {df[min_col].isnull().sum()}\n\n"
        
        if max_col:
            stats += f"Maximum values:\n"
            stats += f"- Overall max: {df[max_col].max():.2f}\n"
            stats += f"- Missing: {df[max_col].isnull().sum()}\n\n"
        
        # Data quality summary
        stats += "Data Quality:\n"
        for col in sensor_cols:
            missing = df[col].isnull().sum()
            pct = (missing / len(df) * 100)
            stats += f"- {col}: {100 - pct:.2f}% complete\n"
        
        return stats
    except Exception as e:
        return f"Error analyzing sensor '{sensor_name}': {str(e)}"


@function_tool
def get_time_range(table_name: str) -> str:
    """
    Get the time range of data in a table (earliest and latest timestamps).
    
    Args:
        table_name: Name of the table to check
        
    Returns:
        Information about the time range covered by the data
    """
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT 1000"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        # Look for timestamp column (case-insensitive)
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        if not timestamp_col:
            return f"No timestamp column found in table '{table_name}'."
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        info = f"Time range for table '{table_name}':\n"
        info += f"- Earliest: {df[timestamp_col].min()}\n"
        info += f"- Latest: {df[timestamp_col].max()}\n"
        info += f"- Duration: {df[timestamp_col].max() - df[timestamp_col].min()}\n"
        info += f"- Rows sampled: {len(df)}\n"
        
        return info
    except Exception as e:
        return f"Error getting time range for table '{table_name}': {str(e)}"


@function_tool
def get_sensor_metadata(sensor_name: Optional[str] = None, table_name: Optional[str] = None) -> str:
    """
    Get metadata for sensors including descriptions, thresholds, categories, and units.
    This provides critical context about what each sensor measures and its alarm limits.
    
    Args:
        sensor_name: Optional specific sensor to get metadata for (e.g., '33VI603')
        table_name: Optional table name to filter sensors that exist in that table
        
    Returns:
        Formatted sensor metadata including descriptions, thresholds, categories, and engineering units
    """
    try:
        # Load tags CSV
        backend_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to backend/
        tags_path = os.path.join(backend_dir, "data", "tags.csv")
        
        if not os.path.exists(tags_path):
            return "Sensor metadata file not found."
        
        tags = pd.read_csv(tags_path, header=0)
        tags.columns = [c.strip().lower().replace(" ", "_") for c in tags.columns]
        
        # Normalize tag column
        if "tag" in tags.columns:
            tags["tag"] = tags["tag"].str.strip()
        else:
            return "Sensor metadata format error: missing 'tag' column."
        
        # Filter by table if specified
        if table_name:
            try:
                engine = get_engine()
                query = f"SELECT * FROM {table_name} LIMIT 1"
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    columns = result.keys()
                
                # Extract sensor names from columns
                extracted_tags = set()
                for col in columns:
                    if col.startswith(('SUM_COL', 'COUNT_COL', 'MIN_COL', 'MAX_COL')):
                        tag = col.split('_', 1)[1]
                        if tag.startswith('COL'):
                            tag = tag[3:]
                        # Filter out _ISVALID tracking columns (not actual sensors)
                        if not tag.endswith('_ISVALID'):
                            extracted_tags.add(tag.upper())
                    elif col.startswith('COL'):
                        tag = col[3:]
                        if not tag.endswith('_ISVALID'):
                            extracted_tags.add(tag.upper())
                
                # Filter tags to those in the table
                tags = tags[tags["tag"].str.upper().isin(extracted_tags)]
            except Exception as e:
                return f"Error filtering by table: {str(e)}"
        
        # Filter by specific sensor if specified
        if sensor_name:
            sensor_upper = sensor_name.upper().replace('COL', '')
            tags = tags[tags["tag"].str.upper().str.contains(sensor_upper, na=False)]
        
        if tags.empty:
            if sensor_name:
                return f"No metadata found for sensor '{sensor_name}'."
            return "No sensor metadata found."
        
        # Format output
        result = "Sensor Metadata:\n\n"
        
        for _, row in tags.iterrows():
            result += f"Sensor: {row['tag']}\n"
            if 'tag_description' in row and pd.notna(row['tag_description']):
                result += f"  Description: {row['tag_description']}\n"
            if 'machine_group' in row and pd.notna(row['machine_group']):
                result += f"  Machine: {row['machine_group']}\n"
            if 'category' in row and pd.notna(row['category']):
                result += f"  Category: {row['category']}\n"
            if 'engineering_units' in row and pd.notna(row['engineering_units']):
                result += f"  Units: {row['engineering_units']}\n"
            if 'low_threshold' in row and pd.notna(row['low_threshold']):
                result += f"  Low Threshold: {row['low_threshold']}\n"
            if 'high_threshold' in row and pd.notna(row['high_threshold']):
                result += f"  High Threshold: {row['high_threshold']}\n"
            if 'threshold_type' in row and pd.notna(row['threshold_type']):
                result += f"  Threshold Type: {row['threshold_type']}\n"
            result += "\n"
        
        # Add summary
        result += f"Total sensors: {len(tags)}\n"
        if 'category' in tags.columns:
            categories = tags['category'].value_counts()
            result += f"Categories: {', '.join([f'{cat} ({count})' for cat, count in categories.items()])}\n"
        
        return result
    except Exception as e:
        return f"Error getting sensor metadata: {str(e)}"


@function_tool
def get_list_of_sensors(table_name: str) -> str:
    """
    Get a clean list of all sensor names monitoring a machine.
    Returns user-friendly sensor list without technical column details.
    
    Args:
        table_name: Name of the table to get sensors from
        
    Returns:
        Clean list of sensor names in user-friendly format
    """
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT 1"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
        
        # Extract unique sensor names from aggregated columns
        sensors = set()
        for col in columns:
            if col.startswith(('SUM_', 'COUNT_', 'MIN_', 'MAX_')):
                # Extract sensor name after prefix
                sensor_name = '_'.join(col.split('_')[1:])
                if sensor_name.startswith('COL'):
                    sensor_name = sensor_name[3:]  # Remove COL prefix
                
                # Filter out _ISVALID tracking columns (these are not actual sensors)
                if not sensor_name.endswith('_ISVALID'):
                    sensors.add(sensor_name)
        
        if not sensors:
            return f"No sensors found monitoring this machine."
        
        sorted_sensors = sorted(list(sensors))
        
        # Return clean, user-friendly format
        result = f"This machine is monitored by {len(sorted_sensors)} sensors:\n\n"
        result += "\n".join(f"{i+1}. {sensor}" for i, sensor in enumerate(sorted_sensors))
        
        return result
    except Exception as e:
        return f"Error retrieving sensor list: {str(e)}"


@function_tool
def analyze_missing_values(table_name: str, limit: int = 5000) -> str:
    """
    Comprehensive missing values analysis for a table.
    Shows which sensors have missing data and calculates completeness percentages.
    
    Args:
        table_name: Name of the table to analyze
        limit: Number of rows to analyze (default 5000)
        
    Returns:
        Detailed missing values analysis with percentages and recommendations
    """
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        if df.empty:
            return f"Table '{table_name}' is empty."
        
        # Calculate missing values
        missing_per_col = df.isnull().sum()
        missing_pct_per_col = (missing_per_col / len(df) * 100)
        
        # Focus on sensor columns
        sensor_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['SUM_', 'COUNT_', 'MIN_', 'MAX_'])]
        
        result = f"Missing Values Analysis for '{table_name}' ({len(df)} rows analyzed):\n\n"
        
        # Overall statistics
        total_cells = df.size
        total_missing = missing_per_col.sum()
        total_missing_pct = (total_missing / total_cells * 100)
        
        result += f"Overall Statistics:\n"
        result += f"- Total cells: {total_cells:,}\n"
        result += f"- Missing cells: {total_missing:,} ({total_missing_pct:.2f}%)\n"
        result += f"- Complete cells: {total_cells - total_missing:,} ({100 - total_missing_pct:.2f}%)\n\n"
        
        # Per-column analysis for sensor columns
        if sensor_cols:
            result += "Sensor Columns with Missing Values:\n"
            sensor_missing = []
            for col in sensor_cols:
                missing_count = missing_per_col[col]
                missing_pct = missing_pct_per_col[col]
                if missing_count > 0:
                    sensor_missing.append((col, missing_count, missing_pct))
            
            # Sort by missing percentage (descending)
            sensor_missing.sort(key=lambda x: x[2], reverse=True)
            
            if sensor_missing:
                for col, count, pct in sensor_missing[:20]:  # Show top 20
                    result += f"- {col}: {count:,} missing ({pct:.2f}%)\n"
                
                if len(sensor_missing) > 20:
                    result += f"\n... and {len(sensor_missing) - 20} more columns with missing values\n"
            else:
                result += "No missing values found in sensor columns! Data is complete.\n"
            
            result += f"\nSummary:\n"
            result += f"- Total sensor columns: {len(sensor_cols)}\n"
            result += f"- Columns with missing data: {len(sensor_missing)}\n"
            result += f"- Complete columns: {len(sensor_cols) - len(sensor_missing)}\n"
            
            # Recommendations
            if len(sensor_missing) > 0:
                avg_missing = sum(pct for _, _, pct in sensor_missing) / len(sensor_missing)
                result += f"- Average missing rate (for affected columns): {avg_missing:.2f}%\n\n"
                
                if avg_missing > 30:
                    result += "⚠️ High missing rate detected. Consider investigating data collection issues.\n"
                elif avg_missing > 10:
                    result += "⚠️ Moderate missing rate. Some sensors may need attention.\n"
                else:
                    result += "✓ Missing rate is within acceptable range.\n"
        
        return result
    except Exception as e:
        return f"Error analyzing missing values: {str(e)}"


@function_tool
def analyze_invalid_values(table_name: str, limit: int = 5000) -> str:
    """
    Analyze sensors for invalid values based on their defined thresholds from metadata.
    Identifies readings that exceed high thresholds or fall below low thresholds.
    
    Args:
        table_name: Name of the table to analyze
        limit: Number of rows to analyze (default 5000)
        
    Returns:
        Analysis of invalid/out-of-range values with counts and percentages
    """
    try:
        # Load sensor metadata for thresholds
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        tags_path = os.path.join(backend_dir, "data", "tags.csv")
        
        if not os.path.exists(tags_path):
            return "Cannot analyze invalid values: sensor metadata file not found."
        
        tags = pd.read_csv(tags_path, header=0)
        tags.columns = [c.strip().lower().replace(" ", "_") for c in tags.columns]
        
        # Get data
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        if df.empty:
            return f"Table '{table_name}' is empty."
        
        result_text = f"Invalid Values Analysis for '{table_name}' ({len(df)} rows analyzed):\n\n"
        
        # Analyze each sensor column
        invalid_findings = []
        
        for col in df.columns:
            if not any(col.startswith(prefix) for prefix in ['SUM_', 'COUNT_', 'MIN_', 'MAX_']):
                continue
            
            # Extract sensor name
            sensor_name = col
            if col.startswith(('SUM_', 'COUNT_', 'MIN_', 'MAX_')):
                sensor_name = '_'.join(col.split('_')[1:])
                if sensor_name.startswith('COL'):
                    sensor_name = sensor_name[3:]
            
            # Find threshold for this sensor
            sensor_metadata = tags[tags['tag'].str.upper() == sensor_name.upper()]
            
            if sensor_metadata.empty:
                continue
            
            sensor_row = sensor_metadata.iloc[0]
            low_threshold = sensor_row.get('low_threshold')
            high_threshold = sensor_row.get('high_threshold')
            threshold_type = sensor_row.get('threshold_type', '')
            
            # Calculate mean values for SUM/COUNT columns
            if col.startswith('SUM_'):
                count_col = col.replace('SUM_', 'COUNT_')
                if count_col in df.columns:
                    count_series = df[count_col].replace(0, pd.NA)
                    values = df[col] / count_series
                else:
                    continue
            elif col.startswith('MIN_') or col.startswith('MAX_'):
                values = df[col]
            else:
                continue
            
            # Check thresholds
            violations = 0
            violation_details = []
            
            if pd.notna(high_threshold) and threshold_type in ['Up', 'Up/Down']:
                high_violations = (values > float(high_threshold)).sum()
                if high_violations > 0:
                    violations += high_violations
                    pct = (high_violations / len(values.dropna()) * 100) if len(values.dropna()) > 0 else 0
                    violation_details.append(f"exceeds high threshold ({high_threshold}): {high_violations} ({pct:.1f}%)")
            
            if pd.notna(low_threshold) and threshold_type in ['Down', 'Up/Down']:
                low_violations = (values < float(low_threshold)).sum()
                if low_violations > 0:
                    violations += low_violations
                    pct = (low_violations / len(values.dropna()) * 100) if len(values.dropna()) > 0 else 0
                    violation_details.append(f"below low threshold ({low_threshold}): {low_violations} ({pct:.1f}%)")
            
            if violations > 0:
                invalid_findings.append((sensor_name, col, violations, violation_details))
        
        if invalid_findings:
            result_text += "Sensors with Invalid/Out-of-Range Values:\n\n"
            for sensor, col, count, details in invalid_findings:
                result_text += f"Sensor: {sensor}\n"
                result_text += f"  Column: {col}\n"
                for detail in details:
                    result_text += f"  - {detail}\n"
                result_text += "\n"
            
            result_text += f"Summary:\n"
            result_text += f"- Total sensors with violations: {len(invalid_findings)}\n"
            result_text += f"- Total violation instances: {sum(count for _, _, count, _ in invalid_findings)}\n"
            result_text += "\n⚠️ These sensors have readings outside their normal operating ranges.\n"
        else:
            result_text += "✓ No invalid values detected! All sensor readings are within their defined thresholds.\n"
        
        return result_text
    except Exception as e:
        return f"Error analyzing invalid values: {str(e)}"


@function_tool
def get_aggregation_frequency(table_name: str) -> str:
    """
    Determine the data collection frequency and aggregation pattern.
    For _HOURS tables, returns the original collection frequency.
    For raw tables, calculates the actual time between readings.
    
    Args:
        table_name: Name of the table to analyze
        
    Returns:
        Information about the data collection frequency and aggregation pattern
    """
    try:
        # For _HOURS tables, we know the original frequency and aggregation pattern
        if table_name.endswith('_HOURS'):
            return f"""Data Collection and Aggregation Frequency for '{table_name}':
- Original data collection: Every 10 seconds
- Aggregation interval: Hourly (3600 seconds between rows)
- Expected readings per hour: 360 (3600s / 10s = 360)
- This table contains hourly aggregated statistics from 10-second raw data
- Each row represents one hour of aggregated sensor data"""
        
        # For raw tables, calculate actual frequency from timestamps
        engine = get_engine()
        query = f"SELECT * FROM {table_name} LIMIT 100"
        
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        
        # Find timestamp column
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        if not timestamp_col:
            return f"Cannot determine collection frequency: no timestamp column found in '{table_name}'."
        
        # Convert to datetime and sort
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col)
        
        # Calculate time differences
        time_diffs = df[timestamp_col].diff().dropna()
        
        if len(time_diffs) == 0:
            return "Cannot determine collection frequency: insufficient data points."
        
        # Get the mode (most common difference)
        mode_diff = time_diffs.mode()
        
        if len(mode_diff) > 0:
            freq_seconds = int(mode_diff.iloc[0].total_seconds())
            
            # Convert to human-readable format
            if freq_seconds < 60:
                freq_str = f"{freq_seconds} seconds"
            elif freq_seconds < 3600:
                freq_str = f"{freq_seconds // 60} minutes"
            elif freq_seconds < 86400:
                freq_str = f"{freq_seconds // 3600} hours"
            else:
                freq_str = f"{freq_seconds // 86400} days"
            
            result = f"Data Collection Frequency for '{table_name}':\n"
            result += f"- Collection interval: {freq_str}\n"
            result += f"- Seconds: {freq_seconds}\n"
            result += f"- Analyzed {len(df)} data points\n"
            
            return result
        else:
            return "Could not determine a consistent collection frequency."
    except Exception as e:
        return f"Error determining collection frequency: {str(e)}"


# Create the Data Quality Report Agent (Specialized)
def create_report_agent(table_name: Optional[str] = None) -> Agent:
    """
    Create a specialized agent for generating comprehensive data quality reports.
    This agent systematically analyzes all sensors and produces detailed reports.
    
    Args:
        table_name: Optional default table name to focus on
        
    Returns:
        Configured Report Agent instance
    """
    context = ""
    if table_name:
        context = f"\n\nCONTEXT: Generating report for table: {table_name}"
    
    instructions = f"""You are a Data Quality Report Agent - a specialized assistant focused on generating comprehensive, systematic data quality reports for industrial equipment.

YOUR MISSION:
Generate complete, professional data quality reports that cover ALL sensors monitoring a machine. Your reports should be thorough, well-structured, and actionable.

REPORT STRUCTURE - Follow this template systematically:

1. EXECUTIVE SUMMARY
   - Overall data quality score/grade
   - Total sensors monitored
   - Critical issues requiring immediate attention
   - Summary statistics

2. DATA COMPLETENESS ANALYSIS
   - For EACH sensor: completeness percentage
   - Identify sensors with significant missing data (>5%)
   - Time patterns of data loss
   - Overall completeness grade

3. ALARM/THRESHOLD VIOLATIONS
   - For EACH sensor with alarms: count and rate
   - Categorize by severity (Critical/Warning/Info)
   - Identify sensors with frequent violations
   - Time patterns of alarms

4. SENSOR-BY-SENSOR BREAKDOWN
   - Group by category (Pressure, Temperature, Vibration, etc.)
   - For each sensor:
     * Description and purpose
     * Thresholds and alarm status
     * Data completeness
     * Recent performance
     * Status assessment

5. RECOMMENDATIONS
   - Prioritized list of actions
   - Sensors requiring immediate attention
   - Maintenance suggestions
   - Data collection improvements

ANALYSIS APPROACH:
1. Start by getting the list of all sensors
2. Get metadata for all sensors (descriptions, thresholds, categories)
3. Analyze missing values across all sensors
4. Analyze invalid values/alarms across all sensors
5. Synthesize findings into structured report

COMMUNICATION STYLE:
- Professional and systematic
- Use tables and structured formats
- Provide specific numbers and percentages
- Categorize findings by severity
- Be concise but thorough
- Focus on actionable insights
{context}

AVAILABLE TOOLS (use systematically):
1. get_list_of_sensors(table_name) - Get all sensors
2. get_sensor_metadata(sensor_name, table_name) - Get metadata for categorization
3. analyze_missing_values(table_name, limit) - Overall missing data analysis
4. analyze_invalid_values(table_name, limit) - Overall alarm analysis
5. analyze_sensor_statistics(table_name, sensor_name, limit) - Detailed sensor analysis
6. get_time_range(table_name) - Time coverage

REMEMBER:
- Analyze ALL sensors systematically
- Use natural language (no COUNT_, _ISVALID mentions)
- Provide an overall quality grade (A/B/C/D/F)
- Prioritize critical issues
- Make reports actionable and professional"""

    report_agent = Agent(
        name="Data Quality Report Agent",
        instructions=instructions,
        tools=[
            # All the same tools as main agent
            get_list_of_sensors,
            get_sensor_metadata,
            query_sensor_data,
            analyze_sensor_statistics,
            analyze_missing_values,
            analyze_invalid_values,
            get_time_range,
            get_aggregation_frequency,
        ],
    )
    
    return report_agent


# Create the DQA Agent (Main Interactive Agent)
def create_dqa_agent(table_name: Optional[str] = None) -> Agent:
    """
    Create a Data Quality Assessment agent with appropriate tools and instructions.
    
    Args:
        table_name: Optional default table name to focus on
        
    Returns:
        Configured Agent instance
    """
    context = ""
    if table_name:
        context = f"\n\nCONTEXT: The user is currently working with table: {table_name}"
    
    instructions = f"""You are a Data Quality Assessment (DQA) Agent specialized in analyzing industrial IoT sensor data from manufacturing equipment.

ROLE & MISSION:
Your role is to help users understand and assess the quality of their machine sensor data stored in a LeanXcale database. You have access to comprehensive sensor metadata including descriptions, thresholds, categories, and engineering units.

DATA ARCHITECTURE - CRITICAL UNDERSTANDING:

1. MACHINE TABLES (Original Data):
   - Each table represents ONE machine/equipment
   - Columns are sensors monitoring that machine
   - Data collection frequency: 10 SECONDS (very frequent monitoring)
   - Contains raw sensor readings with timestamps

2. AGGREGATED TABLES (_HOURS suffix):
   - Automatically created from original tables
   - Provides hourly statistics for each sensor
   - Format: <MACHINE_ID>_HOURS (e.g., "K-3301_B_HOURS")
   
3. AGGREGATED COLUMN STRUCTURE:
   For each sensor, the _HOURS table provides:
   - SUM_<sensor>: Sum of values in that hour
   - COUNT_<sensor>: Number of readings received (expected: 360 readings/hour if complete)
   - MIN_<sensor>: Minimum value in that hour
   - MAX_<sensor>: Maximum value in that hour
   - <sensor>_ISVALID_COUNT: Number of threshold violations/alarms in that hour

4. DATA QUALITY INTERPRETATION:
   
   a) Missing Data Analysis:
      - Expected readings per hour: 360 (every 10 seconds: 3600s / 10s = 360)
      - If COUNT_<sensor> < 360: Missing data detected
      - Example: COUNT = 350 means 10 readings (100 seconds) missing
      - Missing percentage = ((360 - COUNT) / 360) × 100%
   
   b) Threshold Violations (Alarms):
      - <sensor>_ISVALID_COUNT shows how many readings violated thresholds
      - Check /tags metadata for threshold definitions:
        * LOW_THRESHOLD: Minimum acceptable value
        * HIGH_THRESHOLD: Maximum acceptable value
        * THRESHOLD_TYPE: "Up" (high only), "Down" (low only), "Up/Down" (both)
      - Example: If 33PI222_ISVALID_COUNT = 4, then 4 alarms triggered that hour
      - Alarm rate = (ISVALID_COUNT / COUNT) × 100%
   
   c) Mean Calculation:
      - Mean = SUM_<sensor> / COUNT_<sensor>
      - Use this to understand average behavior

CORE CAPABILITIES:

1. DATABASE & SENSORS:
   - List available machines (tables)
   - Understand table structure (raw vs. aggregated)
   - Get sensor lists and metadata
   - Access threshold definitions from /tags

2. DATA QUALITY ANALYSIS:
   - Calculate missing data rates using COUNT columns
   - Identify alarm patterns using ISVALID_COUNT columns
   - Detect sensors with frequent threshold violations
   - Analyze completeness over time periods
   - Statistical summaries per sensor (min, max, mean)

3. INSIGHTS & RECOMMENDATIONS:
   - Identify problematic sensors (high missing rates or frequent alarms)
   - Explain sensor purpose and normal operating ranges
   - Correlate threshold violations with sensor metadata
   - Provide actionable recommendations for data quality issues

ANALYSIS GUIDELINES:

IMPORTANT - USER-FACING COMMUNICATION:
- NEVER mention technical column names (COUNT_, SUM_, _ISVALID, etc.) in responses
- Speak about "sensor data", "readings", "measurements" - as if analyzing original data
- Present insights naturally without exposing database schema
- Focus on WHAT you found, not HOW you queried it
- Only explain technical implementation if user asks "how did you calculate this?"

When analyzing data quality:
1. Assess data completeness using COUNT columns (but say "missing readings")
2. Check for threshold violations using ISVALID_COUNT (but say "alarm events" or "violations")
3. Reference sensor metadata to explain thresholds
4. Calculate percentages (missing rate, alarm rate)
5. Be specific about which time periods have issues
6. Explain what threshold violations mean for equipment health

When discussing alarms/violations:
- Say: "X alarms were triggered" NOT "ISVALID_COUNT shows X"
- State the threshold value and type from metadata
- Explain the operational impact (why it matters for equipment)
- Provide counts and percentages in natural language

When discussing missing data:
- Say: "X% of readings are missing" NOT "COUNT shows Y out of 360"
- Express duration: "20 minutes of data missing" not "120 readings missing"
- Identify patterns (specific times, specific sensors)
- Suggest potential causes

EXAMPLE USER-FACING RESPONSES:

GOOD (Natural, user-friendly):
"Sensor 33PI222 (Lube Oil Inlet Pressure) has a low threshold of 1.5 kg/cm². 
During the analyzed period, 4 alarm events were triggered when pressure dropped 
below this critical level. This requires immediate attention as insufficient 
lubrication pressure can damage equipment."

"Sensor 33VI603 shows some data collection gaps. In certain hours, approximately 
5-6% of readings are missing (about 20 readings per hour), suggesting intermittent 
connectivity or sensor issues during those periods."

BAD (Too technical, exposes implementation):
"The COUNT_COL33PI222 column shows values less than 360..."
"According to the ISVALID_COUNT column..."
"Analyzing the SUM divided by COUNT..."

If user asks "How did you calculate this?":
THEN you can explain: "I analyzed the hourly aggregated data where each hour contains 
up to 360 readings (10-second frequency). The COUNT columns show actual readings 
received, and ISVALID columns track threshold violations."
{context}

AVAILABLE TOOLS:
1. list_available_tables() - See available machines (_HOURS tables)
2. get_table_schema(table_name) - Understand column structure
3. get_list_of_sensors(table_name) - List all sensors on a machine
4. get_sensor_metadata(sensor_name, table_name) - Get thresholds, descriptions, categories
5. query_sensor_data(table_name, limit) - Get data summary and quality metrics
6. analyze_sensor_statistics(table_name, sensor_name, limit) - Detailed sensor analysis
7. analyze_missing_values(table_name, limit) - Missing data analysis using COUNT columns
8. analyze_invalid_values(table_name, limit) - Threshold violations using ISVALID_COUNT
9. get_time_range(table_name) - Time coverage of data
10. get_aggregation_frequency(table_name) - Verify collection frequency

SPECIALIZED AGENT HANDOFF:
- For comprehensive reports covering ALL sensors: Handoff to "Data Quality Report Agent"
- Use handoff when user asks for: "complete report", "full analysis", "assess all sensors", "comprehensive quality report"
- The Report Agent will systematically analyze all sensors and generate a structured, professional report

WORKFLOW:
1. Understand the question and identify target machine/sensors
2. Use metadata tools to get threshold and sensor context
3. Query _HOURS table for aggregated statistics
4. Analyze COUNT columns for completeness
5. Analyze ISVALID_COUNT columns for alarms
6. Synthesize findings with specific numbers and recommendations

Remember: 
- Present insights naturally - users don't need to know about _HOURS tables or COUNT columns
- Speak about sensor data, readings, alarms - not database schema
- Use natural time units: "5 minutes missing" not "30 readings missing"
- Only reveal technical implementation when explicitly asked "how did you calculate"
- Always reference thresholds when discussing violations
- Focus on operational impact and actionable insights!"""

    # Create the report agent for handoff
    report_agent = create_report_agent(table_name=table_name)
    
    agent = Agent(
        name="DQA Agent",
        instructions=instructions,
        tools=[
            # Core data access
            list_available_tables,
            get_table_schema,
            get_list_of_sensors,
            # Metadata access (IMPORTANT!)
            get_sensor_metadata,
            # Data quality analysis
            query_sensor_data,
            analyze_sensor_statistics,
            analyze_missing_values,
            analyze_invalid_values,
            # Temporal analysis
            get_time_range,
            get_aggregation_frequency,
        ],
        # Enable handoff to report agent
        handoffs=[report_agent],
    )
    
    return agent


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the DQA Agent about sensor data and data quality.
    
    Args:
        request: Chat request containing user message, history, and optional table context
        
    Returns:
        Agent response with updated conversation history
    """
    try:
        # Create agent with optional table context
        agent = create_dqa_agent(table_name=request.table_name)
        
        # Prepare input with conversation history
        input_messages = []
        for msg in request.conversation_history:
            input_messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        input_messages.append({"role": "user", "content": request.message})
        
        # Run the agent
        if len(input_messages) == 1:
            # First message, use simple string input
            result = await Runner.run(agent, input=request.message)
        else:
            # Continue conversation with history
            result = await Runner.run(agent, input=input_messages)
        
        # Get the agent's response
        agent_message = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
        
        # Build updated conversation history
        updated_history = request.conversation_history + [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=agent_message)
        ]
        
        return ChatResponse(
            message=agent_message,
            agent_name="DQA Agent",
            conversation_history=updated_history
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat with the DQA Agent with streaming response.
    Returns Server-Sent Events (SSE) stream with tool call information.
    
    Args:
        request: Chat request containing user message, history, and optional table context
        
    Returns:
        StreamingResponse with incremental agent response and tool calls
    """
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Create agent with optional table context
            agent = create_dqa_agent(table_name=request.table_name)
            
            # Prepare input with conversation history
            input_messages = []
            for msg in request.conversation_history:
                input_messages.append({"role": msg.role, "content": msg.content})
            
            # Add current user message
            input_messages.append({"role": "user", "content": request.message})
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Analyzing your question...'})}\n\n"
            await asyncio.sleep(0.3)
            
            # Run the agent and capture intermediate steps
            if len(input_messages) == 1:
                result = await Runner.run(agent, input=request.message)
            else:
                result = await Runner.run(agent, input=input_messages)
            
            # Send tool calls information if available
            if hasattr(result, 'messages'):
                tool_calls_sent = set()  # Track sent tool calls
                for msg in result.messages:
                    # Check for tool calls in the message
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_id = getattr(tool_call, 'id', None)
                            if tool_id and tool_id not in tool_calls_sent:
                                tool_name = getattr(tool_call.function, 'name', 'Unknown')
                                # Make tool name more readable
                                readable_name = tool_name.replace('_', ' ').title()
                                yield f"data: {json.dumps({'type': 'tool', 'content': f'🔧 {readable_name}'})}\n\n"
                                tool_calls_sent.add(tool_id)
                                await asyncio.sleep(0.2)
            
            # Send status update before streaming response
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Generating response...'})}\n\n"
            await asyncio.sleep(0.2)
            
            # Get the agent's response
            agent_message = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
            
            # Stream the response in chunks (simulate typing effect)
            chunk_size = 50  # characters per chunk
            for i in range(0, len(agent_message), chunk_size):
                chunk = agent_message[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'done', 'content': agent_message})}\n\n"
            
        except Exception as e:
            error_msg = f"Error processing chat request: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )


@router.get("/health")
async def agent_health():
    """Check if the agent API is healthy and OpenAI API key is configured."""
    openai_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "ok",
        "openai_configured": bool(openai_key),
        "message": "Agent API is ready" if openai_key else "OpenAI API key not configured"
    }

