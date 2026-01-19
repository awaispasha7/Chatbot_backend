FROM python:3.13-slim

# Install system dependencies required for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libxcb1 \
    libxcb-xinerama0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (including best.pt model file)
COPY . .

# Verify model file exists
RUN if [ ! -f "best.pt" ]; then echo "WARNING: best.pt not found!"; else echo "✅ Model file found: $(du -h best.pt | cut -f1)"; fi

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Start the application
CMD ["python", "start_server.py"]
