"""
FFprobe utility for video validation
"""
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg and ffprobe are installed"""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_duration(file_path: str) -> Optional[float]:
    """
    Get video duration in seconds using ffprobe
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Duration in seconds, or None if failed
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(file_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        return duration
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return None


def get_video_info(file_path: str) -> Optional[dict]:
    """
    Get video information including resolution, codec, fps
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Dictionary with video info, or None if failed
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(file_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": eval(video_stream.get("r_frame_rate", "30/1")),
            "codec": video_stream.get("codec_name"),
            "total_frames": int(video_stream.get("nb_frames", 0)) or None
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, ZeroDivisionError):
        return None


def convert_to_mp4(input_path: str, output_path: str) -> Tuple[bool, str]:
    """
    Convert video to H.264 MP4 using ffmpeg
    
    Args:
        input_path: Path to input video
        output_path: Path for output MP4
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",  # Overwrite output
                str(output_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"
