#!/usr/bin/env python3
"""
Script to add PDF files to the knowledge base
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from simple_knowledge_base import simple_knowledge_base as knowledge_base

def add_pdf_to_knowledge_base(pdf_path: str):
    """Add a PDF file to the knowledge base"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File {pdf_path} not found")
        return False
    
    if not pdf_path.endswith('.pdf'):
        print(f"❌ Error: {pdf_path} is not a PDF file")
        return False
    
    try:
        print(f"📄 Processing PDF: {pdf_path}")
        knowledge_base.add_documents_from_file(pdf_path)
        print(f"✅ Successfully added {pdf_path} to knowledge base")
        return True
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return False

def main():
    """Main function"""
    print("📚 PDF Knowledge Base Uploader")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("Usage: python add_pdf.py <path_to_pdf_file>")
        print("\nExample:")
        print("python add_pdf.py company_projects.pdf")
        print("python add_pdf.py ./documents/recent_proposals.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    # Add PDF to knowledge base
    success = add_pdf_to_knowledge_base(pdf_path)
    
    if success:
        print("\n🎉 PDF successfully added to knowledge base!")
        print("\nThe chatbot can now answer questions about:")
        print("- Your recent projects")
        print("- Company proposals")
        print("- Specific case studies")
        print("- Technical implementations")
        print("\nTest it by asking questions about your projects!")
    else:
        print("\n❌ Failed to add PDF to knowledge base")

if __name__ == "__main__":
    main()
