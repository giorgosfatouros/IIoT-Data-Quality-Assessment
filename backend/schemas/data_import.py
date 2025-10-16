from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ImportStatus(str, Enum):
    """Status of data import job"""
    PENDING = "pending"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    IMPORTING = "importing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MachineType(str, Enum):
    """Supported machine types for import"""
    KT2201 = "KT2201"
    K3301 = "K3301"
    K5700 = "K5700"
    AUTO = "auto"  # Auto-detect from filename

class ImportJobRequest(BaseModel):
    """Request to create a new import job"""
    table_name: str = Field(..., description="Target table name for imported data")
    machine_type: MachineType = Field(default=MachineType.AUTO, description="Machine type (KT2201, K3301, K5700, or auto-detect)")
    overwrite: bool = Field(default=False, description="Whether to drop existing table before import")
    validate_only: bool = Field(default=False, description="Only validate files without importing")
    
    @validator('table_name')
    def validate_table_name(cls, v):
        """Ensure table name is valid SQL identifier"""
        if not v:
            raise ValueError("Table name cannot be empty")
        # Allow alphanumeric and underscores
        if not all(c.isalnum() or c == '_' for c in v):
            raise ValueError("Table name must contain only alphanumeric characters and underscores")
        if v[0].isdigit():
            raise ValueError("Table name cannot start with a digit")
        return v.upper()

class ImportJobResponse(BaseModel):
    """Response after creating import job"""
    job_id: str = Field(..., description="Unique identifier for the import job")
    status: ImportStatus = Field(..., description="Current status of the import job")
    table_name: str = Field(..., description="Target table name")
    machine_type: str = Field(..., description="Detected or specified machine type")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(..., description="Status message")

class ImportJobStatus(BaseModel):
    """Detailed status of an import job"""
    job_id: str
    status: ImportStatus
    table_name: str
    machine_type: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    rows_processed: int = Field(default=0, ge=0)
    rows_total: Optional[int] = None
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

class FileValidationResult(BaseModel):
    """Result of file validation"""
    is_valid: bool
    filename: str
    file_size_bytes: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    detected_machine_type: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)

class TagsValidationResult(BaseModel):
    """Result of tags file validation"""
    is_valid: bool
    filename: str
    file_size_bytes: int
    tag_count: int
    required_columns: List[str]
    missing_columns: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tags_preview: List[Dict[str, Any]] = Field(default_factory=list)

class ImportValidationResponse(BaseModel):
    """Combined validation response"""
    data_file: FileValidationResult
    tags_file: TagsValidationResult
    can_proceed: bool
    suggested_table_name: Optional[str] = None
    suggested_machine_type: Optional[str] = None

class ImportHistoryItem(BaseModel):
    """Historical import job item"""
    job_id: str
    table_name: str
    machine_type: str
    status: ImportStatus
    created_at: datetime
    completed_at: Optional[datetime]
    rows_processed: int
    duration_seconds: Optional[float] = None

class ImportHistoryResponse(BaseModel):
    """Response with import history"""
    jobs: List[ImportHistoryItem]
    total_jobs: int
    successful_jobs: int
    failed_jobs: int

class DockerHealthResponse(BaseModel):
    """Health check response for Docker/moh-importer"""
    docker_available: bool
    importer_image_available: bool
    database_connection: bool
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)



