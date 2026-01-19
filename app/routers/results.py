"""
Results Router - Serves processed video files
"""
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import get_settings
from ..services.job_manager import job_manager
from ..models.schemas import JobStatus

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{job_id}.mp4")
async def get_result_video(job_id: str):
    """
    Serve the processed video result
    
    - Only available after processing is complete
    - Supports range requests for video seeking
    """
    settings = get_settings()
    
    # Check if job exists (synchronous lookup)
    job = job_manager.get_job(job_id)
    
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=404,
            detail=f"Processing not complete. Current status: {job.status}"
        )
    
    # Check if file exists
    output_path = os.path.join(settings.output_dir, f"{job_id}.mp4")
    
    if not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail="Result file not found. It may have been cleaned up."
        )
    
    # Serve the file
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{job.original_filename or 'video'}.mp4"
    )
