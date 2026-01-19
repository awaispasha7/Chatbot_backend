"""
Jobs Router - Handles job status endpoint
"""
from fastapi import APIRouter, HTTPException

from ..models.schemas import JobStatusResponse, ErrorResponse, JobStatus
from ..services.job_manager import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse}
    }
)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    
    - status: queued | processing | done | failed
    - progress: 0-100 percentage
    - error: error message if failed
    - result_url: URL to download result if done
    """
    try:
        # Fast synchronous lookup (no lock needed for read-only access)
        job = job_manager.get_job(job_id)
        
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        result_url = None
        if job.status == JobStatus.DONE:
            result_url = f"/api/results/{job_id}.mp4"
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            result_url=result_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting job status for {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )
