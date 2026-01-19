#!/bin/bash
# Install system dependencies required for OpenCV and FFmpeg
set -e

echo "Installing system dependencies..."

# Update package list
apt-get update

# Install X11 and OpenGL libraries required by OpenCV
apt-get install -y \
    libxcb1 \
    libxcb-xinerama0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxkbcommon-x11-0 \
    libxkbcommon0

# Install FFmpeg for video processing
apt-get install -y ffmpeg

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "System dependencies installed successfully"
