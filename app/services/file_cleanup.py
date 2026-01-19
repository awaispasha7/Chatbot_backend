"""
File Cleanup Service - Removes old uploaded and processed files
"""
import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from ..config import get_settings
from .job_manager import job_manager


async def cleanup_old_files():
    """
    Remove files older than configured TTL
    
    This should be called periodically (e.g., every 30 minutes)
    """
    settings = get_settings()
    ttl = timedelta(hours=settings.file_ttl_hours)
    now = datetime.now()
    
    # Cleanup uploads
    upload_dir = Path(settings.upload_dir)
    if upload_dir.exists():
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if now - file_age > ttl:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up upload: {file_path.name}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
    
    # Cleanup outputs
    output_dir = Path(settings.output_dir)
    if output_dir.exists():
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if now - file_age > ttl:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up output: {file_path.name}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
    
    # Cleanup old jobs from memory
    all_jobs = await job_manager.get_all_jobs()
    for job_id, job in all_jobs.items():
        if now - job.created_at > ttl:
            await job_manager.remove_job(job_id)
            print(f"Cleaned up job: {job_id}")


async def start_cleanup_scheduler():
    """
    Start the periodic cleanup scheduler
    
    Runs cleanup at configured intervals
    """
    settings = get_settings()
    interval = settings.cleanup_interval_minutes * 60  # Convert to seconds
    
    while True:
        await asyncio.sleep(interval)
        try:
            await cleanup_old_files()
        except Exception as e:
            print(f"Cleanup error: {e}")
