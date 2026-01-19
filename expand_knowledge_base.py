#!/usr/bin/env python3
"""
Script to expand the knowledge base with comprehensive agentic AI content
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from simple_knowledge_base import simple_knowledge_base as knowledge_base

def expand_knowledge_base():
    """Expand the knowledge base with comprehensive content"""
    
    print("📚 Expanding Knowledge Base")
    print("=" * 50)
    
    # Check current status
    print("\n1. Current Knowledge Base Status:")
    status = knowledge_base.get_knowledge_base_status()
    print(f"Total Documents: {status['total_documents']}")
    print(f"Document Sources: {status['document_sources']}")
    print(f"Document Types: {status['document_types']}")
    
    # Reinitialize with expanded content
    print("\n2. Adding Comprehensive Agentic AI Content:")
    print("- Advanced agentic AI capabilities")
    print("- Industry-specific solutions")
    print("- Implementation and ROI information")
    print("- Technical architecture details")
    print("- Expanded service offerings")
    
    try:
        # Reinitialize with expanded content
        knowledge_base.initialize_with_agentic_ai_content()
        
        # Check new status
        print("\n3. Updated Knowledge Base Status:")
        new_status = knowledge_base.get_knowledge_base_status()
        print(f"Total Documents: {new_status['total_documents']}")
        print(f"Document Sources: {new_status['document_sources']}")
        print(f"Document Types: {new_status['document_types']}")
        
        print(f"\n✅ Successfully added {new_status['total_documents'] - status['total_documents']} new documents!")
        
    except Exception as e:
        print(f"❌ Error expanding knowledge base: {e}")
        return False
    
    # Test the expanded knowledge base
    print("\n4. Testing Expanded Knowledge Base:")
    test_queries = [
        "What advanced agentic AI capabilities do you offer?",
        "What industries do you serve?",
        "What is your implementation timeline?",
        "What technology stack do you use?",
        "What ROI can I expect?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        try:
            results = knowledge_base.search(query, k=2)
            if results:
                print(f"✅ Found {len(results)} relevant results")
                print(f"Top result: {results[0]['content'][:100]}...")
            else:
                print("❌ No results found")
        except Exception as e:
            print(f"❌ Error testing query: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Knowledge base expansion completed!")
    print("\nYour chatbot now has comprehensive knowledge about:")
    print("- Advanced agentic AI capabilities")
    print("- Industry-specific solutions")
    print("- Implementation processes and timelines")
    print("- Technical architecture and infrastructure")
    print("- ROI and business value propositions")
    print("- Your existing company projects (from uploaded PDF)")
    
    return True

if __name__ == "__main__":
    success = expand_knowledge_base()
    if success:
        print("\n🚀 Ready to test your expanded chatbot!")
        print("Run: python run.py")
        print("Then visit: http://localhost:8000/chat-interface")
    else:
        print("\n❌ Knowledge base expansion failed. Please check the errors above.")
