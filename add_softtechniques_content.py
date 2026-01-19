#!/usr/bin/env python3
"""
Script to add Soft Techniques company content to the knowledge base
"""

from simple_knowledge_base import simple_knowledge_base

def main():
    print("🚀 Adding Soft Techniques Company Content to Knowledge Base")
    print("=" * 50)
    
    try:
        # Add Soft Techniques company content
        simple_knowledge_base.add_softtechniques_company_content()
        
        print("✅ Soft Techniques company content added successfully!")
        print("\n📊 Knowledge Base Status:")
        status = simple_knowledge_base.get_knowledge_base_status()
        print(f"   Total Documents: {status['total_documents']}")
        print(f"   Total Embeddings: {status['total_embeddings']}")
        print(f"   Document Sources: {status['document_sources']}")
        
        print("\n🎉 Your chatbot is now more Soft Techniques company-focused!")
        print("   The chatbot will now emphasize Soft Techniques' custom AI solutions")
        print("   and position agentic AI as one of many services offered.")
        
    except Exception as e:
        print(f"❌ Error adding Soft Techniques content: {e}")

if __name__ == "__main__":
    main()
