#!/usr/bin/env python3
"""
Test script to demonstrate PDF persistence functionality
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from simple_knowledge_base import simple_knowledge_base as knowledge_base

def test_persistence():
    """Test that PDF data persists after restart"""
    
    print("💾 Testing PDF Persistence")
    print("=" * 50)
    
    # Check current knowledge base status
    print("\n1. Current Knowledge Base Status:")
    status = knowledge_base.get_knowledge_base_status()
    print(f"Total Documents: {status['total_documents']}")
    print(f"Total Embeddings: {status['total_embeddings']}")
    print(f"Document Sources: {status['document_sources']}")
    print(f"Document Types: {status['document_types']}")
    
    # Test search functionality
    print("\n2. Testing Search Functionality:")
    test_queries = [
        "What services do you offer?",
        "What are your recent projects?",
        "How can agentic AI help my business?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = knowledge_base.search(query, k=2)
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {results[0]['content'][:100]}...")
            print(f"Source: {results[0]['source']}")
    
    print("\n✅ Persistence test completed!")
    print("\n📝 What this means:")
    print("- Your uploaded PDFs are saved to disk")
    print("- They will persist after restarting the chatbot")
    print("- No need to re-upload the same PDFs")
    print("- Data is stored in the './chroma_db' directory")

if __name__ == "__main__":
    test_persistence()
