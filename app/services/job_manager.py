"""
Job Manager Service - In-memory job state management
For production, replace with Redis or database-backed storage
"""
import asyncio
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field
from ..models.schemas import JobStatus


@dataclass
class Job:
    """Job data structure"""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    error: Optional[str] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    original_filename: Optional[str] = None


class JobManager:
    """
    In-memory job manager for MVP
    
    For production scaling, replace with:
    - Redis for job state
    - Celery/RQ for task queue
    - PostgreSQL for persistence
    """
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(
        self, 
        job_id: str, 
        input_file: str,
        original_filename: str
    ) -> Job:
        """Create a new job entry"""
        async with self._lock:
            job = Job(
                job_id=job_id,
                input_file=input_file,
                original_filename=original_filename
            )
            self._jobs[job_id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID (synchronous for fast access)"""
        return self._jobs.get(job_id)
    
    async def update_status(
        self, 
        job_id: str, 
        status: JobStatus,
        progress: int = None,
        error: str = None,
        output_file: str = None
    ):
        """Update job status and progress"""
        async with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                if progress is not None:
                    job.progress = progress
                if error is not None:
                    job.error = error
                if output_file is not None:
                    job.output_file = output_file
    
    async def get_all_jobs(self) -> Dict[str, Job]:
        """Get all jobs (for cleanup purposes)"""
        return self._jobs.copy()
    
    async def remove_job(self, job_id: str):
        """Remove a job entry"""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]


# Global job manager instance
job_manager = JobManager()
