#!/usr/bin/env python3
"""
Debug script to test consultation detail extraction
"""

import re
from typing import Dict, Optional

def extract_consultation_details(user_message: str) -> Optional[Dict[str, str]]:
    """Extract consultation details from user message"""
    # Look for patterns that indicate consultation details
    consultation_patterns = {
        'name': r'(?:name|i am|my name is|call me)\s*:?\s*([a-zA-Z\s]+)',
        'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        'phone': r'(?:phone|contact|call|number)\s*:?\s*([0-9\s\-\(\)\+]+)',
        'company': r'(?:company|firm|business|organization|work at|work for)\s*:?\s*([a-zA-Z0-9\s&.,]+)',
        'date': r'(?:date|schedule|appointment|meeting)\s*:?\s*([a-zA-Z0-9\s,]+)',
        'time': r'(?:time|at|around)\s*:?\s*([0-9]+\s*(?:am|pm|AM|PM|[0-9:]+))',
        'message': r'(?:message|details|about|regarding|project)\s*:?\s*(.+)'
    }
    
    details = {}
    user_lower = user_message.lower()
    
    print(f"Testing message: '{user_message}'")
    print(f"Lowercase: '{user_lower}'")
    
    # Check if this looks like consultation details
    consultation_indicators = ['schedule', 'consultation', 'appointment', 'meeting', 'call', 'demo']
    has_indicators = any(indicator in user_lower for indicator in consultation_indicators)
    print(f"Has consultation indicators: {has_indicators}")
    print(f"Found indicators: {[indicator for indicator in consultation_indicators if indicator in user_lower]}")
    
    if not has_indicators:
        print("NO consultation indicators found")
        return None
    
    # Extract details using regex patterns
    for field, pattern in consultation_patterns.items():
        matches = re.findall(pattern, user_message, re.IGNORECASE)
        if matches:
            details[field] = matches[0].strip()
            print(f"FOUND {field}: '{details[field]}'")
        else:
            print(f"NO {field} found")
    
    print(f"Extracted details: {details}")
    
    # Must have at least name and email to be considered a consultation request
    if 'name' in details and 'email' in details:
        print("HAS both name and email - would schedule consultation")
        return details
    else:
        print("MISSING name or email - would NOT schedule consultation")
        return None

def test_messages():
    """Test various consultation messages"""
    test_messages = [
        "Hi, I want to schedule a consultation. My name is Shehryar, my email is shahzadashehryar16@gmail.com, my company is Softec Techniques, my phone is 555 2348769, and I'd like to schedule for September 29, 2025 at 7 PM.",
        "I want to schedule a consultation for February 10, 2025 at 5 PM EST. My name is Shehryar and my email is shahzadashehryar16@gmail.com. I work at Meta.",
        "Schedule consultation: Name: John Doe, Email: john@example.com, Company: Acme Corp, Phone: 555-1234, Date: March 15, 2025, Time: 2:00 PM",
        "I need to book a meeting. My name is Jane Smith, email jane@company.com, and I want to discuss AI solutions for my business.",
        "Can I schedule a demo? I'm Mike Johnson from TechCorp, my email is mike@techcorp.com, and I'm interested in your AI services."
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print(f"{'='*60}")
        result = extract_consultation_details(message)
        print(f"Result: {result is not None}")
        print()

if __name__ == "__main__":
    test_messages()
