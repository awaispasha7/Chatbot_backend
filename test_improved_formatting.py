#!/usr/bin/env python3
"""
Test script to verify improved response formatting with proper paragraphs and bullet points
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import rag_system

def test_improved_formatting():
    """Test that responses use proper paragraph and bullet point formatting"""
    
    print("📝 Testing Improved Response Formatting")
    print("=" * 50)
    
    # Test queries that should trigger well-formatted responses
    test_queries = [
        "What is agentic AI?",
        "What services do you offer?",
        "What are the benefits of agentic AI?",
        "How can agentic AI help my business?",
        "What is your implementation process?",
        "What industries do you serve?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 40)
        
        try:
            # Get response
            response = rag_system.chat(query, session_id="formatting_test")
            response_text = response.response
            
            # Analyze formatting structure
            lines = response_text.split('\n')
            has_paragraphs = False
            has_bullet_points = False
            has_line_breaks = False
            
            # Check for proper structure
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-'):
                    has_paragraphs = True
                if line.startswith('-'):
                    has_bullet_points = True
                if line == '':
                    has_line_breaks = True
            
            # Check for markdown issues
            issues = []
            if "**" in response_text:
                issues.append("Contains double asterisks (**)")
            if "*" in response_text and "**" not in response_text:
                issues.append("Contains single asterisks (*)")
            if "###" in response_text or "##" in response_text or "#" in response_text:
                issues.append("Contains markdown headers")
            
            # Display results
            print("📊 Formatting Analysis:")
            if has_paragraphs:
                print("✅ Contains paragraphs")
            else:
                print("❌ Missing paragraphs")
            
            if has_bullet_points:
                print("✅ Contains bullet points")
            else:
                print("❌ Missing bullet points")
            
            if has_line_breaks:
                print("✅ Uses line breaks for structure")
            else:
                print("❌ No line breaks for structure")
            
            if issues:
                print("❌ Formatting Issues:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ No markdown formatting issues")
            
            # Check length
            word_count = len(response_text.split())
            if word_count <= 150:
                print(f"✅ Response length: {word_count} words (under 150)")
            else:
                print(f"❌ Response too long: {word_count} words (over 150)")
            
            # Check for call-to-action
            cta_indicators = ["schedule", "consultation", "contact", "book", "demo", "discuss", "call", "email"]
            has_cta = any(indicator in response_text.lower() for indicator in cta_indicators)
            if has_cta:
                print("✅ Contains call-to-action")
            else:
                print("❌ Missing call-to-action")
            
            # Show formatted response
            print(f"\n📄 Formatted Response:")
            print("-" * 30)
            print(response_text)
            print("-" * 30)
                
        except Exception as e:
            print(f"❌ Error testing query: {e}")
    
    print("\n" + "=" * 50)
    print("📝 Improved formatting test completed!")
    print("\nExpected format structure:")
    print("1. Opening paragraph explaining main point")
    print("2. Bullet points for lists/features/benefits")
    print("3. Closing paragraph with call-to-action")
    print("4. Line breaks to separate sections")
    print("5. No markdown formatting")

def test_specific_formatting_cases():
    """Test specific formatting scenarios"""
    
    print("\n🎯 Testing Specific Formatting Cases")
    print("=" * 50)
    
    # Test cases that should have different formatting
    test_cases = [
        {
            "query": "What is agentic AI?",
            "expected": "Should have opening paragraph + bullet points + closing paragraph"
        },
        {
            "query": "What services do you offer?",
            "expected": "Should have service list with bullet points"
        },
        {
            "query": "How can I contact you?",
            "expected": "Should have contact info with bullet points"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        try:
            response = rag_system.chat(test_case['query'], session_id="specific_test")
            response_text = response.response
            
            # Count different elements
            lines = response_text.split('\n')
            paragraph_lines = [line for line in lines if line.strip() and not line.strip().startswith('-')]
            bullet_lines = [line for line in lines if line.strip().startswith('-')]
            empty_lines = [line for line in lines if not line.strip()]
            
            print(f"📊 Structure Analysis:")
            print(f"- Paragraph lines: {len(paragraph_lines)}")
            print(f"- Bullet point lines: {len(bullet_lines)}")
            print(f"- Empty lines (separators): {len(empty_lines)}")
            
            # Show the response
            print(f"\n📄 Response:")
            print(response_text)
            
        except Exception as e:
            print(f"❌ Error testing case: {e}")

if __name__ == "__main__":
    test_improved_formatting()
    test_specific_formatting_cases()
    
    print("\n🚀 Improved formatting test completed!")
    print("Your chatbot should now provide better structured responses with:")
    print("- Clear opening paragraphs")
    print("- Bullet points for lists and features")
    print("- Proper line breaks for structure")
    print("- Closing paragraphs with calls-to-action")
    print("- No markdown formatting")

