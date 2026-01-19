#!/usr/bin/env python3
"""
Test script to verify proper formatting without markdown
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import rag_system

def test_formatting():
    """Test that responses use proper formatting without markdown"""
    
    print("🎨 Testing Response Formatting")
    print("=" * 50)
    
    # Test queries that should trigger formatted responses
    test_queries = [
        "What is agentic AI?",
        "What services do you offer?",
        "What are the benefits of agentic AI?",
        "How can agentic AI help my business?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query}'")
        print("-" * 40)
        
        try:
            # Get response
            response = rag_system.chat(query, session_id="formatting_test")
            response_text = response.response
            
            # Check for markdown formatting issues
            issues = []
            
            # Check for asterisks
            if "**" in response_text:
                issues.append("Contains double asterisks (**)")
            if "*" in response_text and "**" not in response_text:
                issues.append("Contains single asterisks (*)")
            
            # Check for other markdown
            if "###" in response_text or "##" in response_text or "#" in response_text:
                issues.append("Contains markdown headers")
            
            if "```" in response_text:
                issues.append("Contains code blocks")
            
            # Check for proper bullet points
            has_dashes = "-" in response_text
            has_asterisk_bullets = "* " in response_text
            
            # Display results
            if issues:
                print("❌ Formatting Issues Found:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("✅ No markdown formatting issues")
            
            # Check bullet point format
            if has_dashes and not has_asterisk_bullets:
                print("✅ Uses dashes for bullet points")
            elif has_asterisk_bullets:
                print("❌ Uses asterisks for bullet points (should use dashes)")
            elif has_dashes:
                print("✅ Uses dashes for bullet points")
            
            # Show response preview
            print(f"\nResponse Preview:")
            print(f"'{response_text[:200]}...'")
            
            # Check length
            word_count = len(response_text.split())
            if word_count <= 150:
                print(f"✅ Response length: {word_count} words (under 150)")
            else:
                print(f"❌ Response too long: {word_count} words (over 150)")
            
            # Check for call-to-action
            cta_indicators = ["schedule", "consultation", "contact", "book", "demo", "discuss"]
            has_cta = any(indicator in response_text.lower() for indicator in cta_indicators)
            if has_cta:
                print("✅ Contains call-to-action")
            else:
                print("❌ Missing call-to-action")
                
        except Exception as e:
            print(f"❌ Error testing query: {e}")
    
    print("\n" + "=" * 50)
    print("🎨 Formatting test completed!")
    print("\nExpected format:")
    print("- Clean bullet points with dashes")
    print("- No asterisks or markdown")
    print("- Under 150 words")
    print("- Includes call-to-action")

if __name__ == "__main__":
    test_formatting()
