#!/usr/bin/env python3
"""
Quick start script for the Agentic AI Chatbot
This script handles initialization and starts the server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import fastapi
        import openai
        import langchain
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirements: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("❌ .env file not found")
        print("Please copy env.example to .env and configure your API keys")
        return False
    print("✅ .env file found")
    return True

def main():
    """Main function to start the chatbot"""
    print("🚀 Starting Agentic AI Chatbot...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Initialize the system
    print("🔧 Initializing system...")
    try:
        subprocess.run([sys.executable, "initialize.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Initialization failed")
        sys.exit(1)
    
    # Start the server
    print("🌐 Starting server...")
    print("=" * 50)
    print("Chatbot will be available at:")
    print("  • Web Interface: http://localhost:8000/chat-interface")
    print("  • API Documentation: http://localhost:8000/docs")
    print("  • API Endpoint: http://localhost:8000/chat")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
