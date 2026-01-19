"""
Videos Router - Handles video upload endpoint
"""
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from ..config import get_settings
from ..models.schemas import UploadResponse, ErrorResponse, JobStatus, YouTubeUrlRequest
from ..services.job_manager import job_manager
from ..services.video_processor import process_video_task
from ..utils.ffprobe import get_video_duration
from ..utils.youtube_downloader import download_youtube_video, is_valid_youtube_url, extract_video_id

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse}
    }
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video file for processing
    
    - Validates file extension
    - Validates file size
    - Validates video duration
    - Returns job_id for tracking
    """
    settings = get_settings()
    settings.ensure_directories()
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_filename = f"{job_id}{file_ext}"
    input_path = os.path.join(settings.upload_dir, input_filename)
    
    try:
        # Read and save file with size check
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(input_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                
                # Check size limit
                if total_size > settings.max_file_size_bytes:
                    f.close()
                    os.remove(input_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
                    )
                
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate video duration
    try:
        duration = get_video_duration(input_path)
        if duration is None:
            os.remove(input_path)
            raise HTTPException(
                status_code=400,
                detail="Could not read video file. Ensure it's a valid video format."
            )
        
        if duration > settings.max_duration_seconds:
            os.remove(input_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration: {settings.max_duration_seconds // 60} minutes"
            )
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error validating video duration: {e}", exc_info=True)
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate video file: {str(e)}"
        )
    
    # Create job entry
    try:
        await job_manager.create_job(
            job_id=job_id,
            input_file=input_path,
            original_filename=file.filename
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating job: {e}", exc_info=True)
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create processing job: {str(e)}"
        )
    
    # Queue processing in background
    try:
        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(settings.output_dir, output_filename)
        
        background_tasks.add_task(process_video_task, job_id, input_path, output_path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error queueing processing task: {e}", exc_info=True)
        # Don't fail the request if background task fails - job is already created
    
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Video uploaded successfully. Processing will begin shortly."
    )


@router.post(
    "/youtube",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def process_youtube_url(
    background_tasks: BackgroundTasks,
    request: YouTubeUrlRequest
):
    """
    Process a YouTube video URL
    
    - Validates YouTube URL
    - Downloads video using yt-dlp
    - Validates video duration
    - Returns job_id for tracking
    """
    settings = get_settings()
    settings.ensure_directories()
    
    # Validate YouTube URL
    if not is_valid_youtube_url(request.url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Please provide a valid YouTube video URL."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Download video
    input_filename = f"{job_id}.mp4"
    input_path = os.path.join(settings.upload_dir, input_filename)
    
    try:
        # Run download in executor to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        success, error_msg, duration = await loop.run_in_executor(
            None,
            download_youtube_video,
            request.url,
            input_path,
            settings.max_duration_seconds,
            settings.max_file_size_mb,
            settings.youtube_cookies_path  # Pass cookies path if configured
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=error_msg or "Failed to download YouTube video"
            )
        
        # Validate video duration (double check)
        if duration and duration > settings.max_duration_seconds:
            if os.path.exists(input_path):
                os.remove(input_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration: {settings.max_duration_seconds // 60} minutes"
            )
        
        # Verify file exists and get actual duration
        if not os.path.exists(input_path):
            raise HTTPException(
                status_code=500,
                detail="Downloaded file not found"
            )
        
        # Get actual video duration from file (with retry for file system delays)
        import time
        max_retries = 5
        actual_duration = None
        for attempt in range(max_retries):
            actual_duration = get_video_duration(input_path)
            if actual_duration is not None:
                break
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait 0.5 seconds before retry
        
        if actual_duration is None:
            if os.path.exists(input_path):
                os.remove(input_path)
            raise HTTPException(
                status_code=400,
                detail="Could not read downloaded video file. The file may be corrupted or in an unsupported format."
            )
        
        if actual_duration > settings.max_duration_seconds:
            if os.path.exists(input_path):
                os.remove(input_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Maximum duration: {settings.max_duration_seconds // 60} minutes"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download video: {str(e)}"
        )
    
    # Create job entry FIRST (before returning response)
    video_id = extract_video_id(request.url) or 'youtube'
    job = await job_manager.create_job(
        job_id=job_id,
        input_file=input_path,
        original_filename=f"youtube_{video_id}.mp4"
    )
    
    # Log job creation
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Job created: {job_id}, status: {job.status}")
    print(f"Job created: {job_id}, status: {job.status}", flush=True)
    
    # Queue processing in background
    output_filename = f"{job_id}.mp4"
    output_path = os.path.join(settings.output_dir, output_filename)
    
    background_tasks.add_task(process_video_task, job_id, input_path, output_path)
    
    response = UploadResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="YouTube video downloaded successfully. Processing will begin shortly."
    )
    
    # Log successful response
    logger.info(f"YouTube download completed for job {job_id}")
    print(f"YouTube download completed for job {job_id}", flush=True)
    
    return response
