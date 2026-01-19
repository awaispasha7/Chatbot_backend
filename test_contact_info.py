#!/usr/bin/env python3
"""
Test script to verify contact information is properly integrated
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import rag_system

def test_contact_information():
    """Test that contact information is properly provided"""
    
    print("📞 Testing Contact Information Integration")
    print("=" * 50)
    
    # Test queries that should trigger contact information
    test_queries = [
        "How can I contact you?",
        "What is your phone number?",
        "What is your email address?",
        "I'm interested in your services, how do I get started?",
        "I want to schedule a consultation",
        "What are your contact details?",
        "How can I reach your team?",
        "I need a quote for agentic AI implementation",
        "I'm ready to start my agentic AI project"
    ]
    
    contact_info_found = 0
    total_tests = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 40)
        
        try:
            # Get response
            response = rag_system.chat(query, session_id="contact_test")
            response_text = response.response.lower()
            
            # Check for contact information
            phone_found = "(888) 324-6560" in response_text or "888-324-6560" in response_text
            email_found = "ask@akenotech.com" in response_text
            
            # Display results
            if phone_found and email_found:
                print("✅ Both phone and email provided")
                contact_info_found += 1
            elif phone_found:
                print("✅ Phone number provided")
                contact_info_found += 1
            elif email_found:
                print("✅ Email address provided")
                contact_info_found += 1
            else:
                print("❌ No contact information found")
            
            # Show response preview
            print(f"\nResponse Preview:")
            print(f"'{response.response[:200]}...'")
            
            # Check for call-to-action
            cta_indicators = ["call", "email", "contact", "schedule", "consultation"]
            has_cta = any(indicator in response_text for indicator in cta_indicators)
            if has_cta:
                print("✅ Contains call-to-action")
            else:
                print("❌ Missing call-to-action")
                
        except Exception as e:
            print(f"❌ Error testing query: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Contact Information Test Results:")
    print(f"✅ Contact info provided in {contact_info_found}/{total_tests} responses")
    print(f"📞 Phone: (888) 324-6560")
    print(f"📧 Email: ask@akenotech.com")
    
    if contact_info_found >= total_tests * 0.8:  # 80% success rate
        print("🎉 Contact information integration successful!")
    else:
        print("⚠️ Contact information integration needs improvement")
    
    return contact_info_found >= total_tests * 0.8

def test_interest_detection():
    """Test that contact info is provided when users show interest"""
    
    print("\n🎯 Testing Interest Detection and Contact Provision")
    print("=" * 50)
    
    interest_queries = [
        "I'm interested in implementing agentic AI",
        "This sounds like what my company needs",
        "I want to learn more about your services",
        "How much does this cost?",
        "I need help with automation",
        "My business could benefit from this",
        "I'm looking for AI solutions"
    ]
    
    for i, query in enumerate(interest_queries, 1):
        print(f"\n{i}. Testing Interest Query: '{query}'")
        print("-" * 40)
        
        try:
            response = rag_system.chat(query, session_id="interest_test")
            response_text = response.response.lower()
            
            # Check if contact info is proactively provided
            has_contact = "(888) 324-6560" in response_text or "ask@akenotech.com" in response_text
            has_cta = any(word in response_text for word in ["call", "email", "contact", "schedule"])
            
            if has_contact and has_cta:
                print("✅ Proactively provides contact info and CTA")
            elif has_contact:
                print("✅ Provides contact info")
            elif has_cta:
                print("✅ Provides call-to-action")
            else:
                print("❌ Missing contact info and CTA")
            
            print(f"Response: {response.response[:150]}...")
            
        except Exception as e:
            print(f"❌ Error testing interest query: {e}")

if __name__ == "__main__":
    success = test_contact_information()
    test_interest_detection()
    
    if success:
        print("\n🚀 Contact information integration is working well!")
        print("Your chatbot will now provide contact details when users ask or show interest.")
    else:
        print("\n⚠️ Contact information integration needs attention.")
        print("Check the system prompt and knowledge base content.")
