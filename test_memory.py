#!/usr/bin/env python3
"""
Test script to demonstrate conversation memory functionality
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import rag_system

def test_conversation_memory():
    """Test conversation memory with follow-up questions"""
    
    print("🧠 Testing Conversation Memory")
    print("=" * 50)
    
    session_id = "test_session_123"
    
    # First question
    print("\n1. First Question:")
    print("User: What services do you offer?")
    response1 = rag_system.chat("What services do you offer?", session_id=session_id)
    print(f"Assistant: {response1.response}")
    
    # Follow-up question (should remember the context)
    print("\n2. Follow-up Question:")
    print("User: How much does it cost?")
    response2 = rag_system.chat("How much does it cost?", session_id=session_id)
    print(f"Assistant: {response2.response}")
    
    # Another follow-up
    print("\n3. Another Follow-up:")
    print("User: Can I see a demo?")
    response3 = rag_system.chat("Can I see a demo?", session_id=session_id)
    print(f"Assistant: {response3.response}")
    
    # Test conversation summary
    print("\n4. Conversation Summary:")
    summary = rag_system.get_conversation_summary(session_id)
    print(f"Session ID: {summary['session_id']}")
    print(f"Message Count: {summary['message_count']}")
    print(f"Topics Discussed: {summary['topics_discussed']}")
    print(f"Last Activity: {summary['last_activity'][:50]}...")
    
    print("\n✅ Memory test completed!")

if __name__ == "__main__":
    test_conversation_memory()
