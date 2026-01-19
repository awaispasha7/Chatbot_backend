#!/usr/bin/env python3
"""
Simple test to check consultation detection
"""

import re
from typing import Dict, Optional

def extract_consultation_details(user_message: str) -> Optional[Dict[str, str]]:
    """Extract consultation details from user message"""
    # Look for patterns that indicate consultation details
    consultation_patterns = {
        'name': r'(?:name|i am|my name is|call me)\s*:?\s*([a-zA-Z\s]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
        'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        'phone': r'(?:phone|contact|call|number)\s*:?\s*([0-9\s\-\(\)\+]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
        'company': r'(?:company|firm|business|organization|work at|work for)\s*:?\s*([a-zA-Z0-9\s&.,]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
        'date': r'(?:date|schedule|appointment|meeting)\s*:?\s*([a-zA-Z0-9\s,]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
        'time': r'(?:time|at|around)\s*:?\s*([0-9]+\s*(?:am|pm|AM|PM|[0-9:]+))',
        'message': r'(?:message|details|about|regarding|project)\s*:?\s*(.+)'
    }
    
    details = {}
    user_lower = user_message.lower()
    
    # Check if this looks like consultation details
    consultation_indicators = ['schedule', 'consultation', 'appointment', 'meeting', 'call', 'demo']
    if not any(indicator in user_lower for indicator in consultation_indicators):
        return None
    
    # Extract details using regex patterns
    for field, pattern in consultation_patterns.items():
        matches = re.findall(pattern, user_message, re.IGNORECASE)
        if matches:
            details[field] = matches[0].strip()
    
    # Must have at least name and email to be considered a consultation request
    if 'name' in details and 'email' in details:
        return details
    
    return None

# Test with your actual message
test_message = "I want to schedule a consultation for February 10, 2025 at 5 PM EST. My name is Shehryar and my email is shahzadashehryar16@gmail.com. I work at Meta."

print(f"Testing: {test_message}")
result = extract_consultation_details(test_message)
print(f"Result: {result}")

if result:
    print("SUCCESS: Would schedule consultation")
else:
    print("FAILED: Would NOT schedule consultation")
