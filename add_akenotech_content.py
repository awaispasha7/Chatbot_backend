#!/usr/bin/env python3
"""
Script to add AkenoTech company content to the knowledge base
"""

from simple_knowledge_base import simple_knowledge_base

def main():
    print("🚀 Adding AkenoTech Company Content to Knowledge Base")
    print("=" * 50)
    
    try:
        # Add AkenoTech company content
        simple_knowledge_base.add_akenotech_company_content()
        
        print("✅ AkenoTech company content added successfully!")
        print("\n📊 Knowledge Base Status:")
        status = simple_knowledge_base.get_knowledge_base_status()
        print(f"   Total Documents: {status['total_documents']}")
        print(f"   Total Embeddings: {status['total_embeddings']}")
        print(f"   Document Sources: {status['document_sources']}")
        
        print("\n🎉 Your chatbot is now more AkenoTech company-focused!")
        print("   The chatbot will now emphasize AkenoTech's custom AI solutions")
        print("   and position agentic AI as one of many services offered.")
        
    except Exception as e:
        print(f"❌ Error adding AkenoTech content: {e}")

if __name__ == "__main__":
    main()
