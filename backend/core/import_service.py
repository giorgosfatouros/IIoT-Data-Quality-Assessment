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
    """Service for managing data imports via moh-importer"""
    
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
                 docker_image: str = "moh-importer:latest",
                 db_host: Optional[str] = None,
                 db_port: int = 1529):
        """
        Initialize import service
        
        Args:
            upload_dir: Directory for storing uploaded files
            docker_image: Docker image name for moh-importer
            db_host: Database host (from env or config)
            db_port: Database port
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs_dir = self.upload_dir / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.docker_image = docker_image
        self.db_host = db_host or os.getenv("DATASTORE_HOST", "localhost")
        self.db_port = db_port
        
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
        
        file_size = file_path.stat().st_size
        
        try:
            # Try to read the file
            df = pd.read_csv(file_path, nrows=10)
            
            # Get counts
            with open(file_path, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # Exclude header
            
            column_count = len(df.columns)
            
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
            preview_rows=preview_rows
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
                    # Check for invalid thresholds
                    invalid_thresholds = df[
                        (df['LOW_THRESHOLD'].notna()) & 
                        (df['HIGH_THRESHOLD'].notna()) & 
                        (df['LOW_THRESHOLD'] > df['HIGH_THRESHOLD'])
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
            suggested_machine_type=data_validation.detected_machine_type
        )
    
    def check_docker_health(self) -> Dict[str, Any]:
        """
        Check if Docker and moh-importer image are available
        
        Returns:
            Health check status dictionary
        """
        health = {
            "docker_available": False,
            "importer_image_available": False,
            "database_connection": False,
            "message": "",
            "details": {}
        }
        
        try:
            # Check Docker daemon
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            health["docker_available"] = result.returncode == 0
            
            if not health["docker_available"]:
                health["message"] = "Docker is not available or not running"
                return health
            
            # Check for moh-importer image
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                timeout=5
            )
            health["importer_image_available"] = bool(result.stdout.strip())
            
            if not health["importer_image_available"]:
                health["message"] = f"Docker image '{self.docker_image}' not found. Please build it first."
                health["details"]["build_command"] = f"cd /home/george/moh-importer-main && docker build -t {self.docker_image} ."
                return health
            
            # TODO: Check database connection (optional)
            health["database_connection"] = True  # Assume true for now
            
            health["message"] = "All systems ready"
            
        except subprocess.TimeoutExpired:
            health["message"] = "Docker command timed out"
        except FileNotFoundError:
            health["message"] = "Docker command not found. Is Docker installed?"
        except Exception as e:
            health["message"] = f"Health check error: {str(e)}"
        
        return health
    
    def run_import(self, 
                   job_id: str,
                   data_file_path: Path,
                   tags_file_path: Path,
                   table_name: str,
                   machine_type: str) -> ImportJobStatus:
        """
        Run the import job using Docker
        
        Args:
            job_id: Unique job identifier
            data_file_path: Path to data CSV
            tags_file_path: Path to tags CSV
            table_name: Target table name
            machine_type: Machine type (KT2201, K3301, K5700)
            
        Returns:
            Updated ImportJobStatus
        """
        job = self.jobs[job_id]
        job.status = ImportStatus.IMPORTING
        job.started_at = datetime.now()
        
        try:
            # Create job-specific directory
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Copy files to job directory
            job_data_file = job_dir / "data.csv"
            job_tags_file = job_dir / "tags.csv"
            
            shutil.copy(data_file_path, job_data_file)
            shutil.copy(tags_file_path, job_tags_file)
            
            # Prepare Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",
                "-e", f"FILEPATH=/files/data.csv",
                "-e", f"CONF_FILEPATH=/files/tags.csv",
                "-e", f"TABLE_NAME={machine_type}",
                "-e", f"DATASTORE_HOST={self.db_host}",
                "-e", f"DATASTORE_PORT={self.db_port}",
                "-v", f"{job_dir.absolute()}:/files",
                self.docker_image
            ]
            
            logger.info(f"Running Docker command: {' '.join(docker_cmd)}")
            job.logs.append(f"Starting import: {' '.join(docker_cmd)}")
            
            # Run the import
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Capture logs
            if result.stdout:
                job.logs.extend(result.stdout.split('\n'))
            if result.stderr:
                job.logs.extend(result.stderr.split('\n'))
            
            if result.returncode == 0:
                job.status = ImportStatus.COMPLETED
                job.progress_percentage = 100.0
                job.completed_at = datetime.now()
                
                # Try to extract row count from logs
                for log_line in job.logs:
                    if "Import took" in log_line or "Will flush" in log_line:
                        # Try to parse row count
                        pass
                
                logger.info(f"Import job {job_id} completed successfully")
            else:
                job.status = ImportStatus.FAILED
                job.error_message = f"Import failed with exit code {result.returncode}"
                logger.error(f"Import job {job_id} failed: {job.error_message}")
            
        except subprocess.TimeoutExpired:
            job.status = ImportStatus.FAILED
            job.error_message = "Import timed out after 10 minutes"
            logger.error(f"Import job {job_id} timed out")
        except Exception as e:
            job.status = ImportStatus.FAILED
            job.error_message = f"Import error: {str(e)}"
            logger.error(f"Import job {job_id} error: {str(e)}", exc_info=True)
        
        return job
    
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



