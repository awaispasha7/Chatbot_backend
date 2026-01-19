#!/bin/bash
# Startup script for Render deployment

# Set environment variables
export LANGCHAIN_TRACING_V2=false
export LANGCHAIN_API_KEY=""

# Create necessary directories
mkdir -p uploads
mkdir -p chroma_db

# Start the application
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
