#!/usr/bin/env python3
"""Railway startup script that reads PORT from environment and starts uvicorn"""
import os
import sys

def get_port():
    """Get PORT from environment variable, default to 8000"""
    port = os.getenv("PORT", "8000")
    
    # Extract only numbers from PORT
    port_num = ''.join(filter(str.isdigit, str(port)))
    
    # If no numbers found or empty, use default
    if not port_num:
        print(f"⚠️ WARNING: PORT '{port}' is invalid, using default 8000")
        port_num = "8000"
    
    return int(port_num)

if __name__ == "__main__":
    # Debug: Print environment info
    print("=" * 60)
    print("=== RAILWAY STARTUP DEBUG INFO ===")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"PORT environment variable: {os.getenv('PORT', 'NOT SET')}")
    
    port = get_port()
    print(f"Using PORT: {port}")
    print(f"Checking if main.py exists: {'YES' if os.path.exists('main.py') else 'NO'}")
    print("=" * 60)
    print(f"Starting uvicorn on port {port}...")
    print("=" * 60)
    
    # Start uvicorn with the determined port
    # Use execvp to replace the current process with uvicorn
    os.execvp("uvicorn", [
        "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ])
