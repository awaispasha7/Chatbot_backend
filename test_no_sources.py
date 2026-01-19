#!/usr/bin/env python3
"""
Test script to verify that sources are not included in responses
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import rag_system

def test_no_sources():
    """Test that responses don't include sources"""
    
    print("🔍 Testing No Sources in Responses")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What services do you offer?",
        "How can agentic AI help my business?",
        "What are your recent projects?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 40)
        
        try:
            # Get response
            response = rag_system.chat(query, session_id="test_session")
            
            # Check if sources are mentioned
            response_text = response.response.lower()
            source_indicators = [
                "source", "sources:", "citation", "reference", 
                "[1]", "[2]", "[3]", "according to", "based on"
            ]
            
            found_sources = [indicator for indicator in source_indicators if indicator in response_text]
            
            if found_sources:
                print(f"❌ Found source indicators: {found_sources}")
                print(f"Response: {response.response[:200]}...")
            else:
                print("✅ No source indicators found")
                print(f"Response: {response.response[:200]}...")
            
            # Check if sources field is empty
            if hasattr(response, 'sources') and response.sources:
                print(f"❌ Sources field contains data: {len(response.sources)} sources")
            else:
                print("✅ Sources field is empty or None")
                
        except Exception as e:
            print(f"❌ Error testing query: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Source removal test completed!")
    print("\nThe chatbot should now respond without mentioning sources.")

if __name__ == "__main__":
    test_no_sources()
