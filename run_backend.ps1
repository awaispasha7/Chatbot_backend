# Chatbot Backend - Run Script
# This script activates the virtual environment and starts the FastAPI server

Write-Host "Starting Chatbot Backend..." -ForegroundColor Green

# Navigate to backend directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Check if FFmpeg is available
Write-Host "Checking FFmpeg..." -ForegroundColor Yellow
$ffmpegCheck = & ffmpeg -version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ FFmpeg is installed" -ForegroundColor Green
} else {
    Write-Host "✗ FFmpeg not found in PATH" -ForegroundColor Red
    Write-Host "Please install FFmpeg: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

# Verify Python is from venv
$pythonPath = & .\venv\Scripts\python.exe -c "import sys; print(sys.executable)"
Write-Host "Using Python: $pythonPath" -ForegroundColor Cyan

# Set OPENAI_API_KEY if not set (for local testing)
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = "dummy-key-for-testing"
    Write-Host "Set OPENAI_API_KEY to dummy value for local testing" -ForegroundColor Yellow
}

# Start the server using venv's Python
Write-Host "`nStarting FastAPI server on http://localhost:8000..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Pass environment variable to Python process
$env:OPENAI_API_KEY = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "dummy-key-for-testing" }
& .\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

