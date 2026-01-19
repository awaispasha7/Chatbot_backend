"""
Pydantic schemas for API request/response models
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    """Job processing status enum"""
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Response from video upload endpoint"""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response from job status endpoint"""
    job_id: str
    status: JobStatus
    progress: int = 0
    error: Optional[str] = None
    result_url: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


class YouTubeUrlRequest(BaseModel):
    """Request model for YouTube URL processing"""
    url: str
