"""
Data Import Service
Handles file validation, Docker integration, and import job management
"""

import os
import uuid
import shutil
import subprocess
import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
import logging

from schemas.data_import import (
    ImportStatus, MachineType, ImportJobStatus, 
    FileValidationResult, TagsValidationResult,
    ImportValidationResponse, ImportHistoryItem
)

logger = logging.getLogger(__name__)

class ImportService:
    """Service for managing data imports directly to TimescaleDB"""
    
    # Required columns in tags CSV
    REQUIRED_TAG_COLUMNS = ["TAG", "LOW_THRESHOLD", "HIGH_THRESHOLD", "AGGREGATION_RULE"]
    
    # Machine type detection patterns
    MACHINE_PATTERNS = {
        "KT2201": ["kt2201", "kt-2201", "k-2201"],
        "K3301": ["k3301", "kt3301", "kt-3301", "k-3301"],
        "K5700": ["k5700", "k-5700"]
    }
    
    def __init__(self, 
                 upload_dir: str = "/tmp/fame-uploads",
                 db_host: Optional[str] = None,
                 db_port: int = 5432,
                 batch_size: int = 1000):
        """
        Initialize import service
        
        Args:
            upload_dir: Directory for storing uploaded files
            db_host: Database host (from env or config)
            db_port: Database port (PostgreSQL default: 5432)
            batch_size: Number of rows to insert per batch
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs_dir = self.upload_dir / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.db_host = db_host or os.getenv("DB_HOST") or os.getenv("DB_IP", "localhost")
        self.db_port = db_port
        self.batch_size = batch_size
        
        # In-memory job tracking (in production, use Redis or database)
        self.jobs: Dict[str, ImportJobStatus] = {}
        
    def create_job_id(self) -> str:
        """Generate unique job ID"""
        return f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def detect_machine_type(self, filename: str, preview_data: Optional[pd.DataFrame] = None) -> Optional[str]:
        """
        Detect machine type from filename or data
        
        Args:
            filename: Name of the CSV file
            preview_data: Optional preview of the data
            
        Returns:
            Machine type string or None
        """
        filename_lower = filename.lower()
        
        # Check filename patterns
        for machine_type, patterns in self.MACHINE_PATTERNS.items():
            if any(pattern in filename_lower for pattern in patterns):
                return machine_type
        
        # Check column names if preview data available
        if preview_data is not None and not preview_data.empty:
            columns_str = " ".join(preview_data.columns).lower()
            
            # Check for specific sensor patterns
            if "22si101" in columns_str or "22vi01" in columns_str:
                return "KT2201"
            elif "33vi601" in columns_str or "33ai601" in columns_str:
                return "K3301"
            elif "57vi01" in columns_str:
                return "K5700"
        
        return None
    
    def validate_data_file(self, file_path: Path, filename: str) -> FileValidationResult:
        """
        Validate uploaded data CSV file
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            FileValidationResult with validation details
        """
        errors = []
        warnings = []
        row_count = None
        column_count = None
        preview_rows = []
        detected_machine_type = None
        sensor_tags = []
        
        file_size = file_path.stat().st_size
        
        try:
            # Check if file is in DataSample format
            is_datasample_format = self._is_datasample_format(file_path)
            
            if is_datasample_format:
                # Validate DataSample format
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        
                        # Read row 1: Extract sensor tags
                        header_row = next(reader, None)
                        if not header_row:
                            errors.append("CSV file is empty")
                        else:
                            # Extract sensor tags (skip first 3 columns)
                            original_tags = []
                            for i, col in enumerate(header_row):
                                if i < 3:
                                    continue
                                tag = col.strip()
                                # Only include valid sensor tags (filter out metadata like #CONNECT!)
                                if tag and self._is_valid_sensor_tag(tag):
                                    original_tags.append(tag)
                            
                            if not original_tags:
                                errors.append("No sensor tags found in header row")
                            else:
                                column_count = len(original_tags)
                                # Normalize sensor tags for display (remove .pv suffix)
                                sensor_tags = [self._normalize_sensor_tag(tag) for tag in original_tags]
                            
                            # Read row 2: Skip description row
                            desc_row = next(reader, None)
                            if not desc_row:
                                errors.append("CSV file missing description row")
                            
                            # Validate timestamp column exists at index 3
                            # Check a few data rows
                            data_row_count = 0
                            sample_rows = []
                            for row in reader:
                                data_row_count += 1
                                if len(row) >= 4:
                                    timestamp_str = row[3].strip() if len(row) > 3 else ""
                                    if timestamp_str:
                                        try:
                                            pd.to_datetime(timestamp_str)
                                            # Create preview entry using normalized tags
                                            if len(sample_rows) < 5:
                                                preview_dict = {'timestamp': timestamp_str}
                                                for i, normalized_tag in enumerate(sensor_tags):
                                                    value_col_idx = i + 4
                                                    if value_col_idx < len(row):
                                                        preview_dict[normalized_tag] = row[value_col_idx]
                                                sample_rows.append(preview_dict)
                                        except (ValueError, TypeError):
                                            if data_row_count == 1:
                                                errors.append(f"Invalid timestamp format in row 3: '{timestamp_str}'")
                                
                                if data_row_count >= 10:
                                    break
                            
                            row_count = data_row_count
                            preview_rows = sample_rows
                            
                            # Detect machine type from sensor tags (already normalized)
                            tags_str = " ".join(sensor_tags).lower()
                            if "22si101" in tags_str or "22vi01" in tags_str:
                                detected_machine_type = "KT2201"
                            elif "33vi601" in tags_str or "33ai601" in tags_str:
                                detected_machine_type = "K3301"
                            elif "57vi01" in tags_str:
                                detected_machine_type = "K5700"
                            
                            if not detected_machine_type:
                                warnings.append("Could not auto-detect machine type from sensor tags. Please specify manually.")
                            
                except Exception as e:
                    errors.append(f"DataSample format validation error: {str(e)}")
            else:
                # Validate standard format
                df = pd.read_csv(file_path, nrows=10)
                
                # Get counts
                with open(file_path, 'r') as f:
                    row_count = sum(1 for _ in f) - 1  # Exclude header
                
                column_count = len(df.columns)
                
                # Extract sensor tags (all columns except timestamp)
                timestamp_col = df.columns[0]
                sensor_tags = [self._normalize_sensor_tag(col) for col in df.columns[1:] if col]
                
                # Check for timestamp column
                if df.columns[0].lower() not in ['timestamp', 'time', 'datetime', 'date']:
                    warnings.append("First column should be 'timestamp' - found: " + str(df.columns[0]))
                
                # Validate timestamp format
                try:
                    pd.to_datetime(df.iloc[:, 0])
                except:
                    errors.append("First column contains invalid timestamp format. Expected 'YYYY-MM-DD HH:MM:SS'")
                
                # Check for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    errors.append("No numeric sensor columns found")
                
                # Detect machine type
                detected_machine_type = self.detect_machine_type(filename, df)
                if not detected_machine_type:
                    warnings.append("Could not auto-detect machine type. Please specify manually.")
                
                # Create preview
                preview_rows = df.head(5).to_dict(orient='records')
            
            # Check file size
            if file_size > 500 * 1024 * 1024:  # 500MB
                warnings.append(f"Large file detected ({file_size / 1024 / 1024:.1f}MB). Import may take several minutes.")
            
        except pd.errors.EmptyDataError:
            errors.append("File is empty")
        except pd.errors.ParserError as e:
            errors.append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"File validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return FileValidationResult(
            is_valid=is_valid,
            filename=filename,
            file_size_bytes=file_size,
            row_count=row_count,
            column_count=column_count,
            detected_machine_type=detected_machine_type,
            errors=errors,
            warnings=warnings,
            preview_rows=preview_rows,
            sensor_tags=sensor_tags
        )
    
    def validate_tags_file(self, file_path: Path, filename: str) -> TagsValidationResult:
        """
        Validate tags CSV file
        
        Args:
            file_path: Path to the tags file
            filename: Original filename
            
        Returns:
            TagsValidationResult with validation details
        """
        errors = []
        warnings = []
        tag_count = 0
        tags_preview = []
        missing_columns = []
        
        file_size = file_path.stat().st_size
        
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            columns_upper = [col.upper() for col in df.columns]
            for required_col in self.REQUIRED_TAG_COLUMNS:
                if required_col not in columns_upper:
                    missing_columns.append(required_col)
            
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Normalize column names
                df.columns = [col.upper() for col in df.columns]
                
                # Check TAG column
                if 'TAG' in df.columns:
                    tag_count = len(df['TAG'].dropna())
                    if tag_count == 0:
                        errors.append("TAG column is empty")
                    
                    # Check for duplicates
                    duplicates = df['TAG'].duplicated().sum()
                    if duplicates > 0:
                        warnings.append(f"Found {duplicates} duplicate TAG entries")
                
                # Check threshold columns
                if 'LOW_THRESHOLD' in df.columns and 'HIGH_THRESHOLD' in df.columns:
                    # Convert threshold columns to numeric, handling comma-separated numbers
                    # First, remove commas and convert to numeric
                    low_threshold_numeric = df['LOW_THRESHOLD'].astype(str).str.replace(',', '').apply(
                        lambda x: pd.to_numeric(x, errors='coerce')
                    )
                    high_threshold_numeric = df['HIGH_THRESHOLD'].astype(str).str.replace(',', '').apply(
                        lambda x: pd.to_numeric(x, errors='coerce')
                    )
                    
                    # Check for invalid thresholds (only where both are numeric and not NaN)
                    invalid_thresholds = df[
                        (low_threshold_numeric.notna()) & 
                        (high_threshold_numeric.notna()) & 
                        (low_threshold_numeric > high_threshold_numeric)
                    ]
                    if len(invalid_thresholds) > 0:
                        warnings.append(f"Found {len(invalid_thresholds)} tags where LOW_THRESHOLD > HIGH_THRESHOLD")
                
                # Check AGGREGATION_RULE
                if 'AGGREGATION_RULE' in df.columns:
                    valid_rules = ['min', 'max', 'avg']
                    invalid_rules = df[~df['AGGREGATION_RULE'].str.lower().isin(valid_rules)]['AGGREGATION_RULE'].dropna()
                    if len(invalid_rules) > 0:
                        warnings.append(f"Found invalid AGGREGATION_RULE values (valid: min, max, avg)")
                
                # Create preview
                tags_preview = df.head(10).to_dict(orient='records')
            
        except pd.errors.EmptyDataError:
            errors.append("Tags file is empty")
        except pd.errors.ParserError as e:
            errors.append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"Tags file validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return TagsValidationResult(
            is_valid=is_valid,
            filename=filename,
            file_size_bytes=file_size,
            tag_count=tag_count,
            required_columns=self.REQUIRED_TAG_COLUMNS,
            missing_columns=missing_columns,
            errors=errors,
            warnings=warnings,
            tags_preview=tags_preview
        )
    
    def validate_files(self, 
                      data_file_path: Path, 
                      tags_file_path: Path,
                      data_filename: str,
                      tags_filename: str) -> ImportValidationResponse:
        """
        Validate both data and tags files
        
        Returns:
            ImportValidationResponse with combined validation results
        """
        data_validation = self.validate_data_file(data_file_path, data_filename)
        tags_validation = self.validate_tags_file(tags_file_path, tags_filename)
        
        can_proceed = data_validation.is_valid and tags_validation.is_valid
        
        # Suggest table name based on file or machine type
        suggested_table = None
        if data_validation.detected_machine_type:
            suggested_table = data_validation.detected_machine_type
        else:
            # Use filename without extension
            suggested_table = Path(data_filename).stem.upper()
        
        return ImportValidationResponse(
            data_file=data_validation,
            tags_file=tags_validation,
            can_proceed=can_proceed,
            suggested_table_name=suggested_table,
            suggested_machine_type=data_validation.detected_machine_type,
            available_sensors=data_validation.sensor_tags
        )
    
    def check_docker_health(self) -> Dict[str, Any]:
        """
        Check if database connection is available (renamed from Docker health check)
        
        Returns:
            Health check status dictionary
        """
        health = {
            "docker_available": True,  # Not needed for direct import
            "importer_image_available": True,  # Not needed for direct import
            "database_connection": False,
            "message": "",
            "details": {}
        }
        
        try:
            # Check database connection
            from models.database import get_engine
            engine = get_engine()
            with engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            health["database_connection"] = True
            health["message"] = "Database connection ready"
            
        except Exception as e:
            health["message"] = f"Database connection error: {str(e)}"
            health["details"]["error"] = str(e)
        
        return health
    
    def run_import(self, 
                   job_id: str,
                   data_file_path: Path,
                   tags_file_path: Path,
                   table_name: str,
                   machine_type: str,
                   selected_sensors: Optional[List[str]] = None) -> ImportJobStatus:
        """
        Run the import job by writing directly to TimescaleDB raw_sensor_data table
        
        Args:
            job_id: Unique job identifier
            data_file_path: Path to data CSV
            tags_file_path: Path to tags CSV
            table_name: Target table name (machine group)
            machine_type: Machine type (KT2201, K3301, K5700)
            
        Returns:
            Updated ImportJobStatus
        """
        job = self.jobs[job_id]
        job.status = ImportStatus.IMPORTING
        job.started_at = datetime.now()
        
        try:
            from models.database import get_engine
            from sqlalchemy import text
            
            engine = get_engine()
            job.logs.append(f"Starting import to TimescaleDB for machine group: {table_name}")
            
            # Load tags to get machine_group mapping
            tags_df = pd.read_csv(tags_file_path)
            tags_df.columns = [c.strip().upper() for c in tags_df.columns]
            
            # Get machine_group from tags (should be consistent)
            if 'MACHINE_GROUP' in tags_df.columns:
                machine_group = tags_df['MACHINE_GROUP'].iloc[0] if len(tags_df) > 0 else table_name
            else:
                machine_group = table_name
            
            job.logs.append(f"Machine group: {machine_group}")
            
            # Create a simple threshold lookup from tags (normalize tags for matching)
            threshold_lookup = {}
            if 'TAG' in tags_df.columns and 'LOW_THRESHOLD' in tags_df.columns and 'HIGH_THRESHOLD' in tags_df.columns:
                for _, tag_row in tags_df.iterrows():
                    tag = self._normalize_sensor_tag(str(tag_row['TAG']))
                    
                    # Convert thresholds to float, handling comma-separated numbers
                    low_threshold = None
                    if pd.notna(tag_row['LOW_THRESHOLD']):
                        try:
                            low_str = str(tag_row['LOW_THRESHOLD']).replace(',', '')
                            low_threshold = float(low_str)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid LOW_THRESHOLD for tag {tag}: {tag_row['LOW_THRESHOLD']}")
                            low_threshold = None
                    
                    high_threshold = None
                    if pd.notna(tag_row['HIGH_THRESHOLD']):
                        try:
                            high_str = str(tag_row['HIGH_THRESHOLD']).replace(',', '')
                            high_threshold = float(high_str)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid HIGH_THRESHOLD for tag {tag}: {tag_row['HIGH_THRESHOLD']}")
                            high_threshold = None
                    
                    threshold_lookup[tag] = {
                        'low': low_threshold,
                        'high': high_threshold,
                        'type': str(tag_row.get('THRESHOLD_TYPE', '')).upper() if pd.notna(tag_row.get('THRESHOLD_TYPE')) else None
                    }
            
            # Detect CSV format
            is_datasample_format = self._is_datasample_format(data_file_path)
            
            if is_datasample_format:
                # Use DataSample format parser
                job.logs.append("Detected DataSample format. Parsing CSV...")
                if selected_sensors:
                    job.logs.append(f"Filtering to {len(selected_sensors)} selected sensors: {', '.join(selected_sensors[:10])}{'...' if len(selected_sensors) > 10 else ''}")
                records = self._parse_datasample_csv(data_file_path, selected_sensors=selected_sensors)
                job.logs.append(f"Parsed {len(records)} sensor readings from DataSample format")
                
                # Prepare data for batch insert
                rows_to_insert = []
                total_rows = 0
                total_expected = len(records)
                
                job.logs.append("Processing sensor readings...")
                for record in records:
                    timestamp = record['timestamp']
                    sensor_tag = record['sensor_tag']
                    value = record['value']
                    
                    # Validate value against thresholds
                    quality_flag = 'valid'
                    if sensor_tag in threshold_lookup:
                        threshold = threshold_lookup[sensor_tag]
                        
                        if threshold['type'] == 'DOWN' and threshold['low'] is not None:
                            if value < threshold['low']:
                                quality_flag = 'invalid'
                        elif threshold['type'] == 'UP' and threshold['high'] is not None:
                            if value > threshold['high']:
                                quality_flag = 'invalid'
                        elif threshold['type'] == 'UP/DOWN':
                            if (threshold['low'] is not None and value < threshold['low']) or \
                               (threshold['high'] is not None and value > threshold['high']):
                                quality_flag = 'invalid'
                    
                    rows_to_insert.append({
                        'timestamp': timestamp,
                        'sensor_tag': sensor_tag,
                        'value': value,
                        'machine_group': machine_group,
                        'quality_flag': quality_flag
                    })
                    
                    total_rows += 1
                    
                    # Batch insert when batch size reached
                    if len(rows_to_insert) >= self.batch_size:
                        self._batch_insert_raw_data(engine, rows_to_insert)
                        job.rows_processed = total_rows
                        job.progress_percentage = min(100.0, (total_rows / total_expected * 100))
                        job.logs.append(f"Inserted {total_rows}/{total_expected} rows ({job.progress_percentage:.1f}%)...")
                        rows_to_insert = []
                
                # Insert remaining rows
                if rows_to_insert:
                    self._batch_insert_raw_data(engine, rows_to_insert)
                    total_rows += len(rows_to_insert)
            else:
                # Use standard format parser (existing logic)
                job.logs.append("Detected standard CSV format. Reading data file...")
                data_df = pd.read_csv(data_file_path)
                
                # Ensure first column is timestamp
                if data_df.columns[0].lower() not in ['timestamp', 'time', 'datetime', 'date']:
                    raise ValueError(f"First column must be timestamp, found: {data_df.columns[0]}")
                
                timestamp_col = data_df.columns[0]
                data_df[timestamp_col] = pd.to_datetime(data_df[timestamp_col])
                
                # Get sensor columns (all except timestamp)
                all_sensor_columns = [col for col in data_df.columns if col != timestamp_col]
                
                # Filter to selected sensors if provided
                if selected_sensors:
                    selected_normalized = [self._normalize_sensor_tag(s) for s in selected_sensors]
                    sensor_columns = [
                        col for col in all_sensor_columns
                        if self._normalize_sensor_tag(col) in selected_normalized
                    ]
                    job.logs.append(f"Filtering to {len(sensor_columns)} selected sensors from {len(all_sensor_columns)} available")
                else:
                    sensor_columns = all_sensor_columns
                
                job.logs.append(f"Found {len(sensor_columns)} sensor columns")
                
                # Prepare data for batch insert
                rows_to_insert = []
                total_rows = 0
                total_expected = len(data_df) * len(sensor_columns)
                
                job.logs.append("Processing data rows...")
                for idx, row in data_df.iterrows():
                    timestamp = row[timestamp_col]
                    
                    for sensor_tag_raw in sensor_columns:
                        value = row[sensor_tag_raw]
                        
                        # Skip NaN values
                        if pd.isna(value):
                            continue
                        
                        # Normalize sensor tag
                        sensor_tag = self._normalize_sensor_tag(sensor_tag_raw)
                        if not sensor_tag:
                            continue
                        
                        # Validate value against thresholds
                        quality_flag = 'valid'
                        if sensor_tag in threshold_lookup:
                            threshold = threshold_lookup[sensor_tag]
                            value_float = float(value)
                            
                            if threshold['type'] == 'DOWN' and threshold['low'] is not None:
                                if value_float < threshold['low']:
                                    quality_flag = 'invalid'
                            elif threshold['type'] == 'UP' and threshold['high'] is not None:
                                if value_float > threshold['high']:
                                    quality_flag = 'invalid'
                            elif threshold['type'] == 'UP/DOWN':
                                if (threshold['low'] is not None and value_float < threshold['low']) or \
                                   (threshold['high'] is not None and value_float > threshold['high']):
                                    quality_flag = 'invalid'
                        
                        rows_to_insert.append({
                            'timestamp': timestamp,
                            'sensor_tag': sensor_tag,
                            'value': float(value),
                            'machine_group': machine_group,
                            'quality_flag': quality_flag
                        })
                        
                        total_rows += 1
                        
                        # Batch insert when batch size reached
                        if len(rows_to_insert) >= self.batch_size:
                            self._batch_insert_raw_data(engine, rows_to_insert)
                            job.rows_processed = total_rows
                            job.progress_percentage = min(100.0, (total_rows / total_expected * 100))
                            job.logs.append(f"Inserted {total_rows}/{total_expected} rows ({job.progress_percentage:.1f}%)...")
                            rows_to_insert = []
                
                # Insert remaining rows
                if rows_to_insert:
                    self._batch_insert_raw_data(engine, rows_to_insert)
                    total_rows += len(rows_to_insert)
            
            job.status = ImportStatus.COMPLETED
            job.progress_percentage = 100.0
            job.rows_processed = total_rows
            job.completed_at = datetime.now()
            job.logs.append(f"Import completed successfully. Total rows inserted: {total_rows}")
            
            logger.info(f"Import job {job_id} completed successfully. {total_rows} rows inserted.")
            
        except Exception as e:
            job.status = ImportStatus.FAILED
            job.error_message = f"Import error: {str(e)}"
            job.logs.append(f"ERROR: {str(e)}")
            logger.error(f"Import job {job_id} error: {str(e)}", exc_info=True)
        
        return job
    
    def _is_valid_sensor_tag(self, tag: str) -> bool:
        """
        Check if a tag is a valid sensor tag (not metadata like #CONNECT!)
        
        Args:
            tag: Tag to validate
            
        Returns:
            True if tag appears to be a sensor tag
        """
        if not tag:
            return False
        
        tag_upper = tag.upper().strip()
        
        # Filter out metadata columns
        invalid_patterns = ['#CONNECT', 'CONNECT', 'TIMESTAMP', 'ROWS', 'FROM', 'TO', 'DELTA']
        if any(pattern in tag_upper for pattern in invalid_patterns):
            return False
        
        # Valid sensor tags typically:
        # - Start with numbers (e.g., "75PI808")
        # - Contain letters and numbers
        # - May have .pv suffix
        # - May have underscores (e.g., "32TI448_8")
        tag_clean = tag_upper.rstrip('.PV')
        
        # Must contain at least one digit and one letter
        has_digit = any(c.isdigit() for c in tag_clean)
        has_letter = any(c.isalpha() for c in tag_clean)
        
        # Should start with a digit or letter (not special characters)
        if tag_clean and not (tag_clean[0].isalnum()):
            return False
        
        return has_digit and has_letter
    
    def _normalize_sensor_tag(self, tag: str) -> str:
        """
        Normalize sensor tag by removing .pv suffix and converting to uppercase
        
        Args:
            tag: Sensor tag (e.g., "75PI808.pv" or "75PI808")
            
        Returns:
            Normalized tag (e.g., "75PI808")
        """
        if not tag:
            return ""
        # Remove .pv suffix if present
        tag = tag.upper().rstrip('.PV')
        return tag.strip()
    
    def _parse_datasample_csv(self, file_path: Path, selected_sensors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Parse DataSample format CSV file
        
        Format:
        - Row 1: Header row with sensor tags (e.g., "75PI808.pv", "75PI823.pv")
        - Row 2: Description row (e.g., "75PI808.pv - SnapShot")
        - Row 3+: Data rows where:
          - Columns 1-3: Metadata (Rows:, PHD command:, From:, etc.)
          - Column 4 (index 3): Timestamp (e.g., "1/1/2024 0:00")
          - Columns 5+: Sensor values corresponding to sensor tags from row 1
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries ready for batch insert with keys:
            - timestamp: datetime object
            - sensor_tag: normalized sensor tag (without .pv suffix)
            - value: float sensor value
        """
        records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Read row 1: Extract sensor tags
            header_row = next(reader, None)
            if not header_row:
                raise ValueError("CSV file is empty")
            
            # Extract sensor tags from row 1 (skip first 3 columns and empty columns)
            # Map normalized tag to (original_tag, column_index)
            sensor_tag_indices = {}
            
            for i, col in enumerate(header_row):
                if i < 3:  # Skip first 3 metadata columns
                    continue
                tag = col.strip()
                # Only include valid sensor tags (filter out metadata like #CONNECT!)
                if tag and self._is_valid_sensor_tag(tag):
                    normalized_tag = self._normalize_sensor_tag(tag)
                    sensor_tag_indices[normalized_tag] = (tag, i)
            
            if not sensor_tag_indices:
                raise ValueError("No sensor tags found in header row")
            
            # Filter to selected sensors if provided
            if selected_sensors:
                # Normalize selected sensors for comparison
                selected_normalized = [self._normalize_sensor_tag(s) for s in selected_sensors]
                # Update indices map to only include selected sensors
                sensor_tag_indices = {
                    norm_tag: (orig_tag, idx) 
                    for norm_tag, (orig_tag, idx) in sensor_tag_indices.items()
                    if norm_tag in selected_normalized
                }
                if not sensor_tag_indices:
                    raise ValueError(f"None of the selected sensors were found in the CSV file")
            
            # Read row 2: Skip description row
            desc_row = next(reader, None)
            if not desc_row:
                raise ValueError("CSV file missing description row")
            
            # Process data rows (row 3+)
            row_num = 2  # Track row number for error messages
            for row in reader:
                row_num += 1
                
                if len(row) < 4:
                    # Skip rows that don't have enough columns
                    continue
                
                # Extract timestamp from column index 3
                timestamp_str = row[3].strip() if len(row) > 3 else ""
                if not timestamp_str:
                    continue
                
                try:
                    # Parse timestamp (handles formats like "1/1/2024 0:00")
                    timestamp = pd.to_datetime(timestamp_str)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Row {row_num}: Invalid timestamp '{timestamp_str}': {e}")
                    continue
                
                # Map sensor values to sensor tags using the indices map
                for normalized_tag, (original_tag, col_idx) in sensor_tag_indices.items():
                    # Column index is already the actual column index in the CSV
                    if col_idx >= len(row):
                        continue
                    
                    value_str = row[col_idx].strip() if col_idx < len(row) else ""
                    
                    # Skip empty values
                    if not value_str:
                        continue
                    
                    # Try to convert to float
                    try:
                        value = float(value_str)
                        # Skip NaN values
                        if pd.isna(value):
                            continue
                    except (ValueError, TypeError):
                        # Skip non-numeric values (e.g., "RUNNING")
                        continue
                    
                    if normalized_tag:
                        records.append({
                            'timestamp': timestamp,
                            'sensor_tag': normalized_tag,
                            'value': value
                        })
        
        return records
    
    def _is_datasample_format(self, file_path: Path) -> bool:
        """
        Check if CSV file is in DataSample format
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            True if file appears to be in DataSample format
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if not first_row:
                    return False
                
                # Check if first column contains "Rows:" or similar metadata
                first_col = first_row[0].strip() if len(first_row) > 0 else ""
                if first_col and first_col.lower().startswith('rows'):
                    return True
                
                # Check if first row has many columns and column 3 might be timestamp-like
                if len(first_row) >= 4:
                    # Check if column 3 (index 3) might be a timestamp header
                    third_col = first_row[3].strip() if len(first_row) > 3 else ""
                    if third_col and third_col.lower() in ['timestamp', 'time', 'datetime', 'date']:
                        return True
                
                return False
        except Exception:
            return False
    
    def _batch_insert_raw_data(self, engine, rows: List[Dict]):
        """Insert a batch of rows into raw_sensor_data table"""
        from sqlalchemy import text
        
        if not rows:
            return
        
        # Build INSERT query with proper PostgreSQL syntax
        query = """
            INSERT INTO raw_sensor_data (timestamp, sensor_tag, value, machine_group, quality_flag)
            VALUES (:timestamp, :sensor_tag, :value, :machine_group, :quality_flag)
        """
        
        with engine.connect() as conn:
            # Use executemany for efficient batch insert
            conn.execute(text(query), rows)
            conn.commit()
    
    def get_job_status(self, job_id: str) -> Optional[ImportJobStatus]:
        """Get status of import job"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[ImportJobStatus]:
        """List all import jobs"""
        return list(self.jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running import job
        
        Returns:
            True if cancelled successfully
        """
        job = self.jobs.get(job_id)
        if job and job.status in [ImportStatus.PENDING, ImportStatus.IMPORTING]:
            job.status = ImportStatus.CANCELLED
            job.completed_at = datetime.now()
            # TODO: Kill Docker container if running
            return True
        return False


# Singleton instance
_import_service: Optional[ImportService] = None

def get_import_service() -> ImportService:
    """Get or create import service singleton"""
    global _import_service
    if _import_service is None:
        _import_service = ImportService()
    return _import_service



