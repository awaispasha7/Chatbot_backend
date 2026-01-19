"""
Configuration settings for the Football Analysis Backend
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Pydantic v2 model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra environment variables not defined in Settings
    )
    
    # Model Configuration
    # Default to "./best.pt" for Railway Docker (model is in Chatbot_backend-main/)
    # For local dev, set MODEL_PATH="../best.pt" if model is in project root
    # VideoProcessor will try multiple locations as fallback
    model_path: str = os.getenv("MODEL_PATH", "./best.pt")
    
    # Storage Paths
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    
    # Demo Limits
    max_file_size_mb: int = 500
    max_duration_seconds: int = 300
    
    # Cleanup Settings
    cleanup_interval_minutes: int = 30
    file_ttl_hours: int = 2
    
    # CORS - Updated to include SoftTechniques frontend
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,https://softtechniques.com"
    
    # YouTube Download Settings
    # Optional: Path to cookies file for YouTube downloads (helps bypass bot detection)
    # To get cookies: Use browser extension like "EditThisCookie" or "Get cookies.txt LOCALLY"
    # Export cookies from youtube.com and save as cookies.txt
    youtube_cookies_path: Optional[str] = "./cookies.txt"  # Default to cookies.txt in backend directory
    
    # Allowed video formats
    allowed_extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def ensure_directories(self):
        """Ensure upload and output directories exist"""
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
