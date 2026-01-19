import os
import json
import warnings
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Suppress the Chroma deprecation warning
warnings.filterwarnings("ignore", message=".*Chroma.*deprecated.*", category=DeprecationWarning)
# Simplified imports to avoid langchain_community dependency
try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to basic text processing
    TextLoader = None
    PyPDFLoader = None
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import Config

class KnowledgeBase:
    def __init__(self):
        try:
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI embeddings: {e}")
            print("Note: Set OPENAI_API_KEY environment variable for full functionality")
            # Set to None to handle gracefully - some features may not work
            self.embeddings = None
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vectorstore = None
        if self.embeddings:
            self._initialize_vectorstore()
        else:
            print("Knowledge base initialized without embeddings (chat features may be limited)")
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
            print("Loaded existing vector store")
        except Exception as e:
            print(f"Creating new vector store: {e}")
            self.vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
    
    def add_documents_from_text(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents from text strings"""
        if metadata is None:
            metadata = [{"source": f"text_{i}"} for i in range(len(texts))]
        
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadata)
        ]
        
        chunks = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        print(f"Added {len(chunks)} document chunks to knowledge base")
    
    def add_documents_from_file(self, file_path: str):
        """Add documents from a file"""
        if TextLoader is None or PyPDFLoader is None:
            print("Document loaders not available. Please install langchain-community for file processing.")
            return
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = file_path
        
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        print(f"Added {len(chunks)} chunks from {file_path}")
    
    def search(self, query: str, k: int = Config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown")
            })
        
        return search_results
    
    def initialize_with_agentic_ai_content(self):
        """Initialize knowledge base with agentic AI domain content"""
        agentic_ai_content = [
            {
                "content": """
                Agentic AI refers to artificial intelligence systems that can act autonomously, make decisions, 
                and take actions in pursuit of goals without constant human intervention. These systems are 
                characterized by their ability to reason, plan, and execute tasks in dynamic environments.
                
                Key characteristics of agentic AI include:
                - Autonomous decision-making capabilities
                - Goal-oriented behavior
                - Ability to interact with environments
                - Learning and adaptation mechanisms
                - Multi-step reasoning and planning
                """,
                "metadata": {"source": "agentic_ai_overview", "category": "fundamentals"}
            },
            {
                "content": """
                Agentic AI systems typically consist of several core components:
                
                1. Perception Module: Processes sensory input and environmental data
                2. Reasoning Engine: Performs logical inference and decision-making
                3. Planning System: Generates action sequences to achieve goals
                4. Execution Module: Carries out planned actions
                5. Learning Component: Updates knowledge and strategies based on experience
                6. Memory System: Stores and retrieves relevant information
                
                These components work together to enable autonomous behavior in complex environments.
                """,
                "metadata": {"source": "agentic_ai_architecture", "category": "architecture"}
            },
            {
                "content": """
                Common applications of agentic AI include:
                
                - Autonomous vehicles and robotics
                - Intelligent personal assistants
                - Automated trading systems
                - Smart home automation
                - Industrial process control
                - Healthcare monitoring systems
                - Cybersecurity threat detection
                - Supply chain optimization
                
                These applications benefit from the autonomous, goal-oriented nature of agentic AI systems.
                """,
                "metadata": {"source": "agentic_ai_applications", "category": "applications"}
            },
            {
                "content": """
                Implementing agentic AI systems requires careful consideration of several factors:
                
                Technical Requirements:
                - Robust perception and sensor fusion
                - Efficient reasoning algorithms
                - Scalable planning systems
                - Reliable execution mechanisms
                
                Safety and Ethics:
                - Alignment with human values
                - Robustness and reliability
                - Transparency and explainability
                - Accountability mechanisms
                
                Performance Optimization:
                - Real-time processing capabilities
                - Resource efficiency
                - Scalability considerations
                - Integration with existing systems
                """,
                "metadata": {"source": "agentic_ai_implementation", "category": "implementation"}
            },
            {
                "content": """
                Our company specializes in developing custom agentic AI solutions for businesses across 
                various industries. We offer comprehensive services including:
                
                - Custom agentic AI system development
                - Integration with existing business processes
                - Training and optimization services
                - Ongoing maintenance and support
                - Consultation on AI strategy and implementation
                
                Our team of experts combines deep technical knowledge with practical business experience 
                to deliver solutions that drive real value for our clients.
                """,
                "metadata": {"source": "company_services", "category": "services"}
            }
        ]
        
        texts = [item["content"] for item in agentic_ai_content]
        metadata = [item["metadata"] for item in agentic_ai_content]
        
        self.add_documents_from_text(texts, metadata)
        print("Initialized knowledge base with agentic AI content")

# Initialize knowledge base
knowledge_base = KnowledgeBase()
