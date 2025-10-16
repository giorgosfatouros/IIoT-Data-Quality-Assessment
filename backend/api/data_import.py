"""
Data Import API Router
Handles CSV file uploads and triggers moh-importer via Docker
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional, List
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import logging

from schemas.data_import import (
    ImportJobRequest, ImportJobResponse, ImportJobStatus,
    ImportValidationResponse, ImportHistoryResponse,
    DockerHealthResponse, MachineType, ImportStatus
)
from core.import_service import get_import_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/import", tags=["data-import"])


@router.get("/health", response_model=DockerHealthResponse)
async def check_import_health():
    """
    Check if the import service is ready (Docker available, image built, etc.)
    """
    try:
        service = get_import_service()
        health = service.check_docker_health()
        
        return DockerHealthResponse(
            docker_available=health["docker_available"],
            importer_image_available=health["importer_image_available"],
            database_connection=health["database_connection"],
            message=health["message"],
            details=health.get("details", {})
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return DockerHealthResponse(
            docker_available=False,
            importer_image_available=False,
            database_connection=False,
            message=f"Health check error: {str(e)}",
            details={}
        )


@router.post("/validate", response_model=ImportValidationResponse)
async def validate_upload_files(
    data_file: UploadFile = File(..., description="CSV file containing sensor data"),
    tags_file: UploadFile = File(..., description="CSV file containing sensor tags and thresholds")
):
    """
    Validate uploaded files before import
    
    Checks:
    - File format and structure
    - Required columns presence
    - Data types and formats
    - Machine type detection
    
    Returns validation results and suggestions.
    """
    service = get_import_service()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as data_temp:
        shutil.copyfileobj(data_file.file, data_temp)
        data_temp_path = Path(data_temp.name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tags_temp:
        shutil.copyfileobj(tags_file.file, tags_temp)
        tags_temp_path = Path(tags_temp.name)
    
    try:
        # Validate files
        validation_result = service.validate_files(
            data_file_path=data_temp_path,
            tags_file_path=tags_temp_path,
            data_filename=data_file.filename or "data.csv",
            tags_filename=tags_file.filename or "tags.csv"
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    finally:
        # Clean up temp files
        data_temp_path.unlink(missing_ok=True)
        tags_temp_path.unlink(missing_ok=True)


@router.post("/upload", response_model=ImportJobResponse)
async def upload_and_import(
    background_tasks: BackgroundTasks,
    data_file: UploadFile = File(..., description="CSV file containing sensor data"),
    tags_file: UploadFile = File(..., description="CSV file containing sensor tags and thresholds"),
    table_name: str = Form(..., description="Target table name"),
    machine_type: MachineType = Form(default=MachineType.AUTO, description="Machine type"),
    overwrite: bool = Form(default=False, description="Overwrite existing table"),
    validate_only: bool = Form(default=False, description="Only validate, don't import")
):
    """
    Upload CSV files and start import job
    
    This endpoint:
    1. Validates the uploaded files
    2. Creates an import job
    3. Triggers moh-importer via Docker in the background
    4. Returns job ID for status tracking
    
    The import runs asynchronously - use GET /import/status/{job_id} to track progress.
    """
    service = get_import_service()
    
    # Check health first
    health = service.check_docker_health()
    if not health["docker_available"]:
        raise HTTPException(
            status_code=503,
            detail="Docker is not available. Cannot perform import."
        )
    if not health["importer_image_available"]:
        raise HTTPException(
            status_code=503,
            detail=f"Docker image not found. Please build it first: {health.get('details', {}).get('build_command', 'N/A')}"
        )
    
    # Create job ID
    job_id = service.create_job_id()
    
    # Save uploaded files to job directory
    job_dir = service.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    data_file_path = job_dir / "data.csv"
    tags_file_path = job_dir / "tags.csv"
    
    try:
        # Save files
        with open(data_file_path, "wb") as f:
            shutil.copyfileobj(data_file.file, f)
        
        with open(tags_file_path, "wb") as f:
            shutil.copyfileobj(tags_file.file, f)
        
        # Validate files
        validation = service.validate_files(
            data_file_path=data_file_path,
            tags_file_path=tags_file_path,
            data_filename=data_file.filename or "data.csv",
            tags_filename=tags_file.filename or "tags.csv"
        )
        
        if not validation.can_proceed:
            # Clean up
            shutil.rmtree(job_dir, ignore_errors=True)
            
            errors = validation.data_file.errors + validation.tags_file.errors
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {'; '.join(errors)}"
            )
        
        # Determine machine type
        detected_machine_type = machine_type.value
        if machine_type == MachineType.AUTO:
            detected_machine_type = validation.suggested_machine_type
            if not detected_machine_type:
                raise HTTPException(
                    status_code=400,
                    detail="Could not auto-detect machine type. Please specify manually."
                )
        
        # Normalize table name
        table_name_normalized = table_name.upper()
        
        # Create job status
        job_status = ImportJobStatus(
            job_id=job_id,
            status=ImportStatus.PENDING,
            table_name=table_name_normalized,
            machine_type=detected_machine_type,
            created_at=datetime.now(),
            progress_percentage=0.0,
            rows_processed=0
        )
        
        service.jobs[job_id] = job_status
        
        if validate_only:
            job_status.status = ImportStatus.COMPLETED
            job_status.completed_at = datetime.now()
            message = "Validation successful. Files are ready for import."
        else:
            # Run import in background
            background_tasks.add_task(
                service.run_import,
                job_id=job_id,
                data_file_path=data_file_path,
                tags_file_path=tags_file_path,
                table_name=table_name_normalized,
                machine_type=detected_machine_type
            )
            message = "Import job started. Use the job_id to track progress."
        
        return ImportJobResponse(
            job_id=job_id,
            status=job_status.status,
            table_name=table_name_normalized,
            machine_type=detected_machine_type,
            created_at=job_status.created_at,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        # Clean up on error
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.get("/status/{job_id}", response_model=ImportJobStatus)
async def get_import_status(job_id: str):
    """
    Get detailed status of an import job
    
    Returns current progress, logs, and any errors.
    """
    service = get_import_service()
    
    job = service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Import job {job_id} not found")
    
    return job


@router.get("/jobs", response_model=List[ImportJobStatus])
async def list_import_jobs(
    status: Optional[ImportStatus] = None,
    limit: int = 50
):
    """
    List all import jobs
    
    Optionally filter by status and limit results.
    """
    service = get_import_service()
    
    jobs = service.list_jobs()
    
    # Filter by status if provided
    if status:
        jobs = [job for job in jobs if job.status == status]
    
    # Sort by creation time (newest first)
    jobs = sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    return jobs


@router.get("/history", response_model=ImportHistoryResponse)
async def get_import_history():
    """
    Get import history summary with statistics
    """
    service = get_import_service()
    
    jobs = service.list_jobs()
    
    # Calculate statistics
    total_jobs = len(jobs)
    successful_jobs = sum(1 for job in jobs if job.status == ImportStatus.COMPLETED)
    failed_jobs = sum(1 for job in jobs if job.status == ImportStatus.FAILED)
    
    # Convert to history items
    history_items = []
    for job in sorted(jobs, key=lambda j: j.created_at, reverse=True):
        duration = None
        if job.completed_at and job.started_at:
            duration = (job.completed_at - job.started_at).total_seconds()
        
        history_items.append({
            "job_id": job.job_id,
            "table_name": job.table_name,
            "machine_type": job.machine_type,
            "status": job.status,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "rows_processed": job.rows_processed,
            "duration_seconds": duration
        })
    
    return ImportHistoryResponse(
        jobs=history_items,
        total_jobs=total_jobs,
        successful_jobs=successful_jobs,
        failed_jobs=failed_jobs
    )


@router.post("/cancel/{job_id}")
async def cancel_import_job(job_id: str):
    """
    Cancel a running import job
    
    Attempts to stop the Docker container and mark job as cancelled.
    """
    service = get_import_service()
    
    if service.cancel_job(job_id):
        return {"message": f"Import job {job_id} cancelled successfully"}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job {job_id}. Job may be already completed or not found."
        )


@router.delete("/job/{job_id}")
async def delete_import_job(job_id: str):
    """
    Delete an import job and its associated files
    
    Warning: This permanently removes job data and logs.
    """
    service = get_import_service()
    
    job = service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Import job {job_id} not found")
    
    # Only allow deletion of completed/failed/cancelled jobs
    if job.status in [ImportStatus.PENDING, ImportStatus.IMPORTING]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Cancel it first."
        )
    
    # Remove job directory
    job_dir = service.jobs_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    # Remove from tracking
    del service.jobs[job_id]
    
    return {"message": f"Import job {job_id} deleted successfully"}


@router.get("/machines")
async def list_supported_machines():
    """
    Get list of supported machine types
    """
    return {
        "supported_machines": [
            {
                "type": "KT2201",
                "description": "K-2201/KT-2201 Machine",
                "sensors": ["22SI101", "22VI01", "22VI04", "22VI06", "22VI08", "22PI69", "22PI70", "22ZI10", "22ZI09", "22ZI11"]
            },
            {
                "type": "K3301",
                "description": "K-3301/KT-3301 Machine",
                "sensors": ["33VI601", "33VI602", "33AI601", "33AI602", "33PI222", "33PI601", "33SI501A"]
            },
            {
                "type": "K5700",
                "description": "K-5700 Machine",
                "sensors": ["57VI01", "57VI02", "57VI03", "57VI04"]
            }
        ],
        "auto_detection": "Supported via filename or column pattern matching"
    }



