@echo off
REM Chatbot Backend - Run Script (Windows Batch)
echo Starting Chatbot Backend...

REM Navigate to backend directory
cd /d "%~dp0"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check FFmpeg
echo Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] FFmpeg is installed
) else (
    echo [WARNING] FFmpeg not found in PATH
    echo Please install FFmpeg: https://ffmpeg.org/download.html
)

REM Verify Python is from venv
echo Using Python from venv...
venv\Scripts\python.exe --version

REM Set OPENAI_API_KEY if not set (for local testing)
if "%OPENAI_API_KEY%"=="" (
    set OPENAI_API_KEY=dummy-key-for-testing
    echo Set OPENAI_API_KEY to dummy value for local testing
)

REM Start the server using venv's Python
echo.
echo Starting FastAPI server on http://localhost:8000...
echo Press Ctrl+C to stop the server
echo.

venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause

