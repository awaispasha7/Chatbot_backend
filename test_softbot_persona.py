#!/usr/bin/env python3
"""
Test script to verify the new Softbot persona
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_softbot_persona():
    """Test the new Softbot persona"""
    print("🧪 Testing Softbot Persona...")
    
    test_messages = [
        "Hi, who are you?",
        "What services do you offer?",
        "Tell me about your company",
        "How can you help my business?"
    ]
    
    for message in test_messages:
        try:
            print(f"\n📤 Testing: '{message}'")
            response = requests.post(f"{BASE_URL}/chat", json={
                "message": message,
                "session_id": "test_softbot_persona"
            })
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['response']
                print(f"✅ Response: {response_text[:150]}...")
                
                # Check if response mentions Softbot or Soft Techniques
                if "softbot" in response_text.lower() or "soft techniques" in response_text.lower():
                    print("   ✅ Mentions Softbot/Soft Techniques")
                else:
                    print("   ⚠️  Doesn't mention Softbot/Soft Techniques")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

def main():
    """Run the test"""
    print("🚀 Testing Softbot Persona")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not running. Please start the chatbot first:")
            print("   python run.py")
            return
    except:
        print("❌ Cannot connect to server. Please start the chatbot first:")
        print("   python run.py")
        return
    
    print("✅ Server is running!")
    
    # Test the persona
    test_softbot_persona()
    
    print("\n" + "=" * 40)
    print("🎉 Softbot persona test completed!")
    print("\n📝 Expected behavior:")
    print("   - Chatbot should introduce itself as Softbot from Soft Techniques")
    print("   - Responses should be personable and professional")
    print("   - Should emphasize Soft Techniques' custom AI solutions")

if __name__ == "__main__":
    main()

