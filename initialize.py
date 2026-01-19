#!/usr/bin/env python3
"""
Initialization script for the Agentic AI Chatbot
This script sets up the knowledge base and initializes the system
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from simple_knowledge_base import simple_knowledge_base as knowledge_base
from config import Config

def check_environment():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment configuration...")
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not getattr(Config, var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def initialize_knowledge_base():
    """Initialize the knowledge base with default content"""
    print("📚 Initializing knowledge base...")
    
    try:
        knowledge_base.initialize_with_agentic_ai_content()
        print("✅ Knowledge base initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")
        return False

def test_knowledge_base():
    """Test the knowledge base with sample queries"""
    print("🧪 Testing knowledge base...")
    
    test_queries = [
        "What is agentic AI?",
        "What services do you offer?",
        "How can agentic AI benefit my business?"
    ]
    
    for query in test_queries:
        try:
            results = knowledge_base.search(query, k=3)
            print(f"✅ Query '{query}' returned {len(results)} results")
            if results:
                print(f"   Top result score: {results[0]['score']:.3f}")
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
            return False
    
    print("✅ Knowledge base tests passed!")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating necessary directories...")
    
    directories = [
        Config.CHROMA_PERSIST_DIRECTORY,
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main initialization function"""
    print("🚀 Initializing Agentic AI Chatbot...")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Initialize knowledge base
    if not initialize_knowledge_base():
        sys.exit(1)
    
    # Test knowledge base
    if not test_knowledge_base():
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Initialization completed successfully!")
    print("\nNext steps:")
    print("1. Copy env.example to .env and configure your API keys")
    print("2. Run: python main.py")
    print("3. Use the API at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
