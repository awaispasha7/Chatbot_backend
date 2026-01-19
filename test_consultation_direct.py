#!/usr/bin/env python3
"""
Direct test of consultation detection and scheduling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import rag_system

def test_consultation_detection():
    """Test consultation detection directly"""
    
    # Test message similar to what you sent
    test_message = "I want to schedule a consultation for February 10, 2025 at 5 PM EST. My name is Shehryar and my email is shahzadashehryar16@gmail.com. I work at Meta."
    
    print(f"Testing message: '{test_message}'")
    print("=" * 60)
    
    # Test consultation detection
    consultation_details = rag_system._extract_consultation_details(test_message)
    print(f"Consultation details detected: {consultation_details}")
    
    if consultation_details:
        print("✅ Consultation details detected - would schedule")
        
        # Test scheduling (but don't actually call the API)
        print("Would call scheduling API with:")
        for key, value in consultation_details.items():
            print(f"  {key}: {value}")
    else:
        print("❌ No consultation details detected - would NOT schedule")
    
    print("\n" + "=" * 60)
    
    # Test the full response generation
    print("Testing full response generation...")
    try:
        response = rag_system.generate_response(
            query=test_message,
            context="",
            session_id="test_session"
        )
        
        print(f"Response generated: {response.response[:200]}...")
        print(f"Response length: {len(response.response)}")
        
        # Check if it's a consultation response or regular response
        if "consultation ID" in response.response.lower() or "successfully scheduled" in response.response.lower():
            print("✅ Generated consultation scheduling response")
        else:
            print("❌ Generated regular response (not consultation scheduling)")
            
    except Exception as e:
        print(f"❌ Error generating response: {e}")

if __name__ == "__main__":
    test_consultation_detection()
