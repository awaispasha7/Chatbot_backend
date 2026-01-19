"""
YouTube Video Downloader Utility
Uses yt-dlp to download videos from YouTube
"""
import os
import re
import logging
from pathlib import Path
from typing import Optional, Tuple
import yt_dlp

logger = logging.getLogger(__name__)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validate if the URL is a valid YouTube URL
    
    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID (YouTube Shorts)
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - URLs with additional parameters (e.g., &t=, &list=, &si=, etc.)
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid YouTube URL
    """
    if not url or not isinstance(url, str):
        return False
    
    # Strip whitespace
    url = url.strip()
    
    # More flexible patterns that handle URLs with query parameters
    youtube_patterns = [
        # Standard watch URLs: youtube.com/watch?v=VIDEO_ID (with optional params)
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # Short URLs: youtu.be/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        # YouTube Shorts: youtube.com/shorts/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        # Embed URLs: youtube.com/embed/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        # Legacy v URLs: youtube.com/v/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        # Mobile URLs: m.youtube.com/watch?v=VIDEO_ID
        r'(?:https?://)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # Mobile Shorts: m.youtube.com/shorts/VIDEO_ID
        r'(?:https?://)?m\.youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    # Use search instead of match to find pattern anywhere in the string
    # This allows URLs with additional parameters or text before/after
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    return False


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL
    
    Supports various YouTube URL formats and handles URLs with query parameters.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID or None if invalid
    """
    if not url or not isinstance(url, str):
        return None
    
    # Strip whitespace
    url = url.strip()
    
    # Patterns that match various YouTube URL formats
    patterns = [
        # Standard watch URLs: youtube.com/watch?v=VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # Short URLs: youtu.be/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        # YouTube Shorts: youtube.com/shorts/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        # Embed URLs: youtube.com/embed/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        # Legacy v URLs: youtube.com/v/VIDEO_ID
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        # Mobile URLs: m.youtube.com/watch?v=VIDEO_ID
        r'(?:https?://)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        # Mobile Shorts: m.youtube.com/shorts/VIDEO_ID
        r'(?:https?://)?m\.youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def normalize_youtube_url(url: str) -> str:
    """
    Normalize YouTube URL to standard watch format
    Converts Shorts URLs and other formats to standard watch?v= format
    This can help bypass some YouTube bot detection issues
    
    Args:
        url: YouTube URL in any format
        
    Returns:
        Normalized URL in watch?v= format, or original URL if conversion fails
    """
    if not url or not isinstance(url, str):
        return url
    
    url = url.strip()
    video_id = extract_video_id(url)
    
    if video_id:
        # Convert to standard watch URL format
        return f"https://www.youtube.com/watch?v={video_id}"
    
    return url


def download_youtube_video(
    url: str,
    output_path: str,
    max_duration_seconds: int = 300,
    max_file_size_mb: int = 500,
    cookies_path: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Download YouTube video using yt-dlp
    
    Args:
        url: YouTube video URL
        output_path: Path to save the downloaded video
        max_duration_seconds: Maximum video duration in seconds
        max_file_size_mb: Maximum file size in MB
        cookies_path: Optional path to cookies file (helps bypass YouTube bot detection)
        
    Returns:
        Tuple of (success, error_message, duration)
    """
    if not is_valid_youtube_url(url):
        return False, "Invalid YouTube URL", None
    
    # Normalize URL (convert Shorts URLs to watch format for better compatibility)
    normalized_url = normalize_youtube_url(url)
    if normalized_url != url:
        logger.info(f"Normalized YouTube URL from {url} to {normalized_url}")
        print(f"Normalized YouTube URL from {url} to {normalized_url}", flush=True)
    url = normalized_url
    print(f"Downloading YouTube video: {url}", flush=True)
    
    # Resolve cookies path to absolute path (important when running in executor)
    print(f"DEBUG: cookies_path received: {cookies_path}", flush=True)
    if cookies_path:
        # If relative path, resolve relative to backend directory (where cookies.txt should be)
        if not os.path.isabs(cookies_path):
            # Get backend directory (parent of app directory)
            backend_dir = Path(__file__).parent.parent.parent
            cookies_path = str(backend_dir / cookies_path)
        # Verify file exists
        if not os.path.exists(cookies_path):
            logger.warning(f"Cookies file not found at: {cookies_path}")
            print(f"WARNING: Cookies file not found at: {cookies_path}", flush=True)
            cookies_path = None
        else:
            logger.info(f"Using cookies file: {cookies_path}")
            print(f"✓ Using cookies file: {cookies_path}", flush=True)
    else:
        print("WARNING: No cookies path provided", flush=True)
    
    try:
        # Configure yt-dlp options
        # Use a temporary filename without extension, yt-dlp will add the extension
        base_output = output_path.rsplit('.', 1)[0]  # Remove .mp4 extension
        
        # Try different YouTube clients to bypass bot detection
        # Order: tv_embedded -> ios -> android -> mweb -> web -> None
        # tv_embedded sometimes works better for regular videos
        clients_to_try = ['tv_embedded', 'ios', 'android', 'mweb', 'web', None]  # None = default client
        last_error = None
        duration = None
        download_success = False
        
        import time
        
        # Add initial delay to avoid immediate rate limiting
        time.sleep(1)
        
        for idx, client in enumerate(clients_to_try):
            try:
                client_name = client if client else 'default'
                print(f"Attempting download with yt-dlp client: {client_name}", flush=True)
                logger.info(f"Attempting download with yt-dlp client: {client_name}")
                
                # Add delay between attempts to avoid rate limiting (longer delay for later attempts)
                if idx > 0:
                    delay = min(3 + idx, 6)  # 3s, 4s, 5s, 6s, 6s, 6s
                    logger.info(f"Waiting {delay} seconds before trying next client with client '{client_name}'...")
                    print(f"Waiting {delay} seconds before trying next client with client '{client_name}'...", flush=True)
                    time.sleep(delay)
                
                # Configure yt-dlp with options to bypass YouTube bot detection
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'outtmpl': f'{base_output}.%(ext)s',
                    'quiet': False,
                    'no_warnings': False,
                    'noplaylist': True,
                    'max_filesize': max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
                    # Bypass YouTube bot detection with different clients
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    # Additional headers to mimic a real browser - including Referer is important
                    'http_headers': {
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Referer': 'https://www.youtube.com/',
                        'Origin': 'https://www.youtube.com',
                        'Connection': 'keep-alive',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Upgrade-Insecure-Requests': '1',
                    },
                    # Retry options for network issues
                    'retries': 3,
                    'fragment_retries': 3,
                    'ignoreerrors': False,
                }
                
                # Add cookies if provided (helps bypass YouTube bot detection)
                if cookies_path and os.path.exists(cookies_path):
                    ydl_opts['cookies'] = cookies_path
                    logger.info(f"Using cookies file: {cookies_path}")
                    print(f"✓ Adding cookies to yt-dlp: {cookies_path}", flush=True)
                elif cookies_path:
                    logger.warning(f"Cookies file specified but not found: {cookies_path}")
                    print(f"WARNING: Cookies file specified but not found: {cookies_path}", flush=True)
                
                # Only add extractor_args if client is specified
                if client is not None:
                    ydl_opts['extractor_args'] = {
                        'youtube': {
                            'player_client': [client],  # Try different clients
                        }
                    }
                else:
                    # Try without specifying client (uses yt-dlp default)
                    logger.info("Trying with default yt-dlp client (no specific client specified)")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        # Get video info first to check duration
                        info = ydl.extract_info(url, download=False)
                        duration = info.get('duration', 0)
                        
                        if duration > max_duration_seconds:
                            return False, f"Video too long. Maximum duration: {max_duration_seconds // 60} minutes", None
                        
                        # Download the video
                        ydl.download([url])
                        
                        # If we get here, download succeeded
                        download_success = True
                        client_name = client if client else 'default'
                        logger.info(f"Successfully downloaded using {client_name} client")
                        print(f"✓ Successfully downloaded YouTube video using {client_name} client: {url}", flush=True)
                        break
                    except Exception as inner_e:
                        # Catch any exception during extraction/download and treat as client failure
                        error_msg = str(inner_e)
                        raise yt_dlp.utils.DownloadError(error_msg)
                    
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                last_error = error_msg
                client_name = client if client else 'default'
                
                # Check for different types of errors
                is_bot_detection = "Sign in to confirm" in error_msg or "not a bot" in error_msg.lower() or "cookies" in error_msg.lower()
                is_extraction_error = (
                    "Failed to extract" in error_msg or 
                    "unable to extract" in error_msg.lower() or 
                    "parse JSON" in error_msg or
                    "JSONDecodeError" in error_msg or
                    "Failed to parse JSON" in error_msg or
                    "Expecting value" in error_msg  # JSON parsing error indicator
                )
                
                # Log the error
                error_type = "bot detection" if is_bot_detection else ("extraction error" if is_extraction_error else "unknown error")
                logger.warning(f"yt-dlp download failed with client '{client_name}': {error_msg[:200]}")
                print(f"WARNING: yt-dlp download failed with client '{client_name}' ({error_type}): {error_msg[:200]}", flush=True)
                
                # If it's bot detection or extraction error, try next client
                if is_bot_detection or is_extraction_error:
                    logger.info(f"{error_type} with {client_name} client, trying next client...")
                    if client == clients_to_try[-1]:  # Last client failed
                        print(f"ERROR: All clients failed. Last error: {error_msg[:200]}", flush=True)
                        raise  # Re-raise to be handled by outer exception handler
                    continue  # Try next client
                else:
                    # Different error, re-raise
                    print(f"ERROR: Unexpected error with {client_name} client: {error_msg[:200]}", flush=True)
                    raise
        
        if not download_success:
            if last_error:
                # Check if it's a bot detection error
                if "Sign in to confirm" in last_error or "not a bot" in last_error.lower() or "cookies" in last_error.lower():
                    error_message = "YouTube is blocking automated downloads for this video. This is a YouTube-side restriction, not a problem with our system. Recommended solution: Use the 'Upload File' option - download the video manually from YouTube and upload it directly. Alternative: Try a different video (some videos work, some don't due to YouTube's bot detection)."
                elif "Failed to extract" in last_error or "unable to extract" in last_error.lower():
                    error_message = "Unable to extract video information from YouTube. The video may be private, age-restricted, or unavailable. Try a different video or use the 'Upload File' option."
                else:
                    error_message = f"Failed to download YouTube video: {last_error[:200]}"
                
                print(f"ERROR: Failed to download YouTube video after multiple attempts: {url}. Last error: {last_error[:200]}", flush=True)
                logger.error(f"Failed to download YouTube video after multiple attempts: {url}. Last error: {last_error}")
                return False, error_message, None
            else:
                error_message = "Failed to download YouTube video. Please try again or use the 'Upload File' option."
                print(f"ERROR: Failed to download YouTube video with all clients: {url}", flush=True)
                logger.error(f"Failed to download YouTube video with all clients: {url}")
                return False, error_message, None
        
        # Find the downloaded file (yt-dlp adds extension)
        downloaded_file = None
        for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
            potential_file = f'{base_output}{ext}'
            if os.path.exists(potential_file):
                downloaded_file = potential_file
                break
        
        if not downloaded_file:
            return False, "Downloaded file not found", None
        
        # If the file is not MP4, rename it to the expected output path
        # The video processor will handle conversion if needed
        if downloaded_file != output_path:
            if downloaded_file.endswith('.mp4'):
                os.rename(downloaded_file, output_path)
            else:
                # Rename to output_path, conversion will happen in video processor
                os.rename(downloaded_file, output_path)
        
        return True, None, duration
            
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        # Clean up ANSI color codes from error messages
        import re
        error_msg = re.sub(r'\x1b\[[0-9;]*m', '', error_msg)  # Remove ANSI codes
        
        if "Private video" in error_msg:
            return False, "This video is private or unavailable", None
        elif "Video unavailable" in error_msg:
            return False, "Video is unavailable or has been removed", None
        elif "Sign in to confirm your age" in error_msg or "age verification" in error_msg.lower():
            return False, "This video requires age verification", None
        elif "Sign in to confirm" in error_msg or "not a bot" in error_msg.lower() or "cookies" in error_msg.lower():
            # YouTube bot detection - all clients failed
            logger.warning(f"YouTube bot detection triggered for all clients: {error_msg[:200]}")
            return False, "YouTube is blocking automated downloads for this video. Some videos cannot be downloaded automatically due to YouTube's bot detection. Please try: 1) A different video, 2) Wait a few minutes and try again, or 3) Use the file upload option instead.", None
        elif "Failed to extract" in error_msg or "unable to extract" in error_msg.lower() or "parse JSON" in error_msg or "JSONDecodeError" in error_msg or "Failed to parse JSON" in error_msg or "Expecting value" in error_msg:
            # YouTube API/parsing error - YouTube is blocking requests completely
            # This often happens with YouTube Shorts or videos with strict bot detection
            logger.warning(f"YouTube extraction error (all clients failed): {error_msg[:200]}")
            return False, "YouTube is blocking automated downloads for this video. This is a YouTube-side restriction, not a problem with our system. Recommended solution: Use the 'Upload File' option - download the video manually from YouTube and upload it directly. Alternative: Try a different video (some videos work, some don't due to YouTube's bot detection).", None
        else:
            # Clean error message for user display
            clean_error = error_msg.split('\n')[0]  # Get first line
            clean_error = clean_error[:200]  # Limit length
            return False, f"Download failed: {clean_error}", None
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {e}", exc_info=True)
        error_msg = str(e)
        # Provide more user-friendly error messages
        if "HTTP Error" in error_msg or "URLError" in error_msg:
            return False, "Network error while downloading video. Please check your internet connection.", None
        elif "ffmpeg" in error_msg.lower() or "codec" in error_msg.lower():
            return False, "Video format not supported. Please try a different video.", None
        else:
            return False, f"Failed to download video: {error_msg[:100]}", None
