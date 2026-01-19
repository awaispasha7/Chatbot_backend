"""
Simplified Knowledge Base without langchain_community dependency
This version uses only basic LangChain components
"""

import os
import json
import pickle
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from config import Config

# Persistent vector store with file-based storage
class SimpleVectorStore:
    def __init__(self, embeddings, persist_directory="./vector_store"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.documents = []
        self.embeddings_cache = {}
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Load existing data
        self._load_from_disk()
    
    def add_documents(self, documents):
        """Add documents to the store"""
        for doc in documents:
            self.documents.append(doc)
            # Create embedding for the document
            embedding = self.embeddings.embed_query(doc.page_content)
            self.embeddings_cache[doc.page_content] = embedding
        
        # Persist after adding documents
        self.persist()
    
    def similarity_search_with_score(self, query, k=5):
        """Simple similarity search"""
        if not self.documents:
            return []
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities (simple cosine similarity)
        similarities = []
        for doc in self.documents:
            if doc.page_content in self.embeddings_cache:
                doc_embedding = self.embeddings_cache[doc.page_content]
                # Simple dot product similarity
                similarity = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                similarities.append((doc, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def persist(self):
        """Persist the store to disk"""
        try:
            # Save documents
            documents_file = os.path.join(self.persist_directory, "documents.pkl")
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings cache
            embeddings_file = os.path.join(self.persist_directory, "embeddings.pkl")
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            
            print(f"Vector store persisted to {self.persist_directory}")
            
        except Exception as e:
            print(f"Error persisting vector store: {e}")
    
    def _load_from_disk(self):
        """Load existing data from disk"""
        try:
            # Load documents
            documents_file = os.path.join(self.persist_directory, "documents.pkl")
            if os.path.exists(documents_file):
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from disk")
            
            # Load embeddings cache
            embeddings_file = os.path.join(self.persist_directory, "embeddings.pkl")
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.embeddings_cache)} embeddings from disk")
                
        except Exception as e:
            print(f"Error loading vector store from disk: {e}")
            # Initialize empty if loading fails
            self.documents = []
            self.embeddings_cache = {}

class SimpleKnowledgeBase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        # Use persistent vector store
        self.vectorstore = SimpleVectorStore(
            self.embeddings, 
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY
        )
        print("Initialized persistent knowledge base")
    
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
        """Add documents from a file (supports PDF and text files)"""
        try:
            if file_path.endswith('.pdf'):
                # Handle PDF files
                import PyPDF2
                content = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text() + "\n"
                
                if not content.strip():
                    print(f"Warning: No text extracted from PDF {file_path}")
                    return
                    
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Create document
            document = Document(
                page_content=content,
                metadata={"source": file_path, "type": "company_document"}
            )
            
            chunks = self.text_splitter.split_documents([document])
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
            print(f"Added {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
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
        """Initialize knowledge base with comprehensive agentic AI domain content"""
        agentic_ai_content = [
            {
                "content": """
                What is Agentic AI?
                
                Agentic AI is autonomous artificial intelligence that makes decisions and takes actions without constant human oversight. Think of it as AI that can think, plan, and execute tasks independently. Unlike traditional AI that requires human input for every decision, agentic AI can analyze situations, make choices, and take actions on its own.
                
                Key Characteristics of Agentic AI:
                - Autonomous Decision Making - Can make complex decisions without human intervention
                - Goal-Oriented Behavior - Works towards specific objectives
                - Learning and Adaptation - Improves performance over time
                - Multi-Task Capability - Can handle multiple tasks simultaneously
                - Context Awareness - Understands and responds to changing environments
                
                Why Choose Our Agentic AI Services?
                - 24/7 Autonomous Operations - Your AI works around the clock
                - Reduced Human Intervention - Lower operational costs
                - Faster Decision Making - Real-time responses to business needs
                - Scalable Solutions - Grows with your business
                - Custom Implementation - Tailored to your specific requirements
                
                Ready to transform your business with autonomous AI? Contact us for a free consultation to see how agentic AI can revolutionize your operations.
                """,
                "metadata": {"source": "agentic_ai_overview", "category": "fundamentals"}
            },
            {
                "content": """
                Our Agentic AI Architecture
                
                We build custom agentic AI systems with these proven components:
                
                Core Modules:
                - Smart Perception - Processes data from multiple sources
                - Advanced Reasoning - Makes intelligent decisions
                - Strategic Planning - Creates action plans
                - Automated Execution - Carries out tasks independently
                - Continuous Learning - Improves over time
                
                Business Benefits:
                - Seamless Integration - Works with your existing systems
                - Custom Development - Tailored to your specific needs
                - Ongoing Support - We maintain and optimize your AI
                
                Interested in a custom agentic AI solution? Schedule a demo to see our architecture in action.
                """,
                "metadata": {"source": "agentic_ai_architecture", "category": "architecture"}
            },
            {
                "content": """
                Agentic AI Applications We Build
                
                Popular Use Cases:
                - Customer Service Automation - 24/7 intelligent support
                - Business Process Automation - Streamline operations
                - Data Analysis & Insights - Automated reporting
                - Supply Chain Optimization - Smart logistics
                - Cybersecurity Monitoring - Threat detection
                - Financial Trading Systems - Automated decisions
                
                Industry Solutions:
                - Healthcare - Patient monitoring systems
                - Manufacturing - Quality control automation
                - Retail - Inventory management
                - Finance - Risk assessment
                
                Ready to automate your business processes? Let's discuss which application would benefit your company most.
                """,
                "metadata": {"source": "agentic_ai_applications", "category": "applications"}
            },
            {
                "content": """
                Our Implementation Process
                
                How We Build Your Agentic AI:
                - Discovery Phase - We analyze your business needs
                - Custom Development - Build AI tailored to your processes
                - Integration - Seamlessly connect with your systems
                - Testing & Optimization - Ensure peak performance
                - Deployment & Support - Launch and maintain your AI
                
                What You Get:
                - Dedicated Project Manager - Personal support throughout
                - Custom Training - Your team learns to use the system
                - 24/7 Monitoring - We watch your AI's performance
                - Ongoing Updates - Continuous improvements
                
                Timeline: Most projects completed in 8-12 weeks
                
                Ready to start your agentic AI journey? Book a consultation to discuss your specific requirements.
                """,
                "metadata": {"source": "agentic_ai_implementation", "category": "implementation"}
            },
            {
                "content": """
                Our Agentic AI Services
                
                What We Offer:
                - Custom AI Development - Built specifically for your business
                - System Integration - Works with your existing tools
                - Training & Support - Your team becomes AI-ready
                - Ongoing Maintenance - We keep your AI running smoothly
                - Strategic Consulting - AI roadmap for your business
                
                Why Choose Us:
                - Proven Track Record - 100+ successful AI implementations
                - Expert Team - PhD-level AI specialists
                - Industry Experience - We understand your business challenges
                - ROI Focused - We measure success by your results
                
                Industries We Serve:
                Healthcare, Finance, Manufacturing, Retail, Logistics, and more
                
                Ready to transform your business with AI? Contact us today for a free consultation and see how we can help you achieve your goals.
                """,
                "metadata": {"source": "company_services", "category": "services"}
            },
            {
                "content": """
                Advanced Agentic AI Capabilities
                
                Multi-Agent Systems:
                - Collaborative AI agents that work together to solve complex problems
                - Distributed decision-making across multiple specialized agents
                - Agent communication protocols and coordination mechanisms
                - Swarm intelligence for large-scale optimization problems
                
                Autonomous Learning and Adaptation:
                - Continuous learning from new data and experiences
                - Self-improving algorithms that get better over time
                - Transfer learning across different domains and tasks
                - Meta-learning for rapid adaptation to new environments
                
                Advanced Reasoning and Planning:
                - Causal reasoning and counterfactual analysis
                - Multi-step planning with uncertainty handling
                - Goal decomposition and hierarchical task planning
                - Dynamic replanning based on changing conditions
                
                Human-AI Collaboration:
                - Natural language interfaces for seamless interaction
                - Explainable AI that provides clear reasoning
                - Human-in-the-loop systems for critical decisions
                - Augmented intelligence that enhances human capabilities
                
                Real-World Applications:
                - Autonomous vehicles and transportation systems
                - Smart cities and urban planning optimization
                - Scientific discovery and research automation
                - Creative content generation and design assistance
                
                Interested in implementing advanced agentic AI capabilities? Let's discuss how these technologies can benefit your specific use case.
                """,
                "metadata": {"source": "advanced_agentic_ai", "category": "advanced_capabilities"}
            },
            {
                "content": """
                Industry-Specific Agentic AI Solutions
                
                Healthcare and Life Sciences:
                - Medical diagnosis assistance and treatment recommendation systems
                - Drug discovery and molecular design automation
                - Patient monitoring and predictive health analytics
                - Medical imaging analysis and radiology assistance
                - Clinical trial optimization and patient recruitment
                
                Financial Services:
                - Algorithmic trading and portfolio management
                - Fraud detection and risk assessment systems
                - Credit scoring and loan approval automation
                - Regulatory compliance monitoring and reporting
                - Customer service and financial advisory chatbots
                
                Manufacturing and Supply Chain:
                - Predictive maintenance and equipment optimization
                - Quality control and defect detection systems
                - Supply chain optimization and demand forecasting
                - Production planning and resource allocation
                - Inventory management and logistics optimization
                
                Retail and E-commerce:
                - Personalized product recommendations
                - Dynamic pricing and revenue optimization
                - Customer behavior analysis and segmentation
                - Inventory management and demand forecasting
                - Chatbot and virtual shopping assistants
                
                Technology and Software:
                - Code generation and software development assistance
                - Automated testing and quality assurance
                - DevOps automation and infrastructure management
                - Cybersecurity threat detection and response
                - API development and integration automation
                
                Each solution is customized to your industry's specific requirements and regulatory environment. Contact us to discuss your industry needs.
                """,
                "metadata": {"source": "industry_solutions", "category": "industry_specific"}
            },
            {
                "content": """
                Agentic AI Implementation and ROI
                
                Implementation Timeline:
                - Week 1-2: Discovery and requirements analysis
                - Week 3-4: Solution design and architecture planning
                - Week 5-8: Development and initial testing
                - Week 9-10: Integration and system testing
                - Week 11-12: Deployment and team training
                
                Expected ROI and Benefits:
                - 30-50% reduction in operational costs through automation
                - 40-60% improvement in process efficiency
                - 25-35% increase in customer satisfaction scores
                - 50-70% reduction in manual errors and processing time
                - 20-40% increase in revenue through better decision-making
                
                Cost Structure:
                - Initial development and setup costs
                - Monthly maintenance and support fees
                - Training and onboarding costs
                - Integration and customization fees
                - Ongoing optimization and enhancement costs
                
                Success Metrics We Track:
                - Process automation percentage
                - Error reduction rates
                - Time-to-completion improvements
                - Customer satisfaction scores
                - Revenue impact and cost savings
                - System uptime and reliability
                
                Our clients typically see ROI within 6-12 months of implementation. Ready to calculate your potential ROI? Schedule a consultation to discuss your specific use case.
                """,
                "metadata": {"source": "implementation_roi", "category": "business_value"}
            },
            {
                "content": """
                Technical Architecture and Infrastructure
                
                Core Technology Stack:
                - Machine Learning Frameworks: TensorFlow, PyTorch, Scikit-learn
                - Natural Language Processing: GPT models, BERT, Transformers
                - Computer Vision: OpenCV, YOLO, ResNet, Vision Transformers
                - Reinforcement Learning: OpenAI Gym, Stable Baselines, Ray RLLib
                - Vector Databases: Pinecone, Weaviate, ChromaDB for semantic search
                
                Cloud and Infrastructure:
                - Cloud Platforms: AWS, Google Cloud, Azure, hybrid solutions
                - Containerization: Docker, Kubernetes for scalable deployment
                - API Management: RESTful APIs, GraphQL, microservices architecture
                - Data Processing: Apache Spark, Kafka for real-time data streaming
                - Monitoring: Prometheus, Grafana for system monitoring and alerting
                
                Security and Compliance:
                - Data encryption at rest and in transit
                - GDPR, HIPAA, SOX compliance frameworks
                - Role-based access control and authentication
                - Audit logging and compliance reporting
                - Secure model deployment and versioning
                
                Integration Capabilities:
                - REST APIs and webhook integrations
                - Database connectors for major systems
                - Third-party service integrations
                - Legacy system modernization
                - Real-time data synchronization
                
                Scalability and Performance:
                - Horizontal scaling for high-volume processing
                - Load balancing and auto-scaling capabilities
                - Caching strategies for improved performance
                - Edge computing for low-latency applications
                - Multi-region deployment for global availability
                
                Our architecture is designed for enterprise-scale deployment with 99.9% uptime guarantee. Contact us to discuss your technical requirements.
                """,
                "metadata": {"source": "technical_architecture", "category": "technology"}
            },
            {
                "content": """
                Contact Information and Getting Started
                
                Ready to transform your business with agentic AI? Here's how to get in touch with our team:
                
                Official Contact Details:
                - Phone: (888) 324-6560
                - Email: ask@akenotech.com
                
                How to Reach Us:
                - Call us directly for immediate assistance and consultation
                - Email us for detailed project inquiries and proposals
                - Schedule a free consultation to discuss your specific needs
                - Get a personalized quote for your agentic AI implementation
                
                What to Expect:
                - Free initial consultation to understand your requirements
                - Custom solution design tailored to your business
                - Detailed project timeline and implementation plan
                - Transparent pricing and ROI projections
                - Ongoing support and maintenance services
                
                When to Contact Us:
                - You're interested in implementing agentic AI in your business
                - You want to learn more about our services and capabilities
                - You need a consultation for your specific use case
                - You're ready to start your agentic AI journey
                - You have questions about our implementation process
                
                Our team of AI experts is ready to help you leverage the power of autonomous artificial intelligence. Contact us today to get started!
                """,
                "metadata": {"source": "contact_information", "category": "contact"}
            }
        ]
        
        texts = [item["content"] for item in agentic_ai_content]
        metadata = [item["metadata"] for item in agentic_ai_content]
        
        self.add_documents_from_text(texts, metadata)
        print("Initialized knowledge base with agentic AI content")
    
    def add_softtechniques_company_content(self):
        """Add Soft Techniques company-focused content to the knowledge base"""
        print("Adding Soft Techniques company content to knowledge base...")
        
        softtechniques_documents = [
            {
                "content": """
                About Soft Techniques - Custom AI Solutions Company
                
                Soft Techniques is a leading custom AI solutions company that specializes in developing tailored artificial intelligence systems for businesses across various industries. We combine cutting-edge AI technology with deep industry expertise to deliver solutions that drive real business value.
                
                Our Mission:
                - Transform businesses through custom AI solutions
                - Deliver practical, business-focused AI implementations
                - Build long-term partnerships with our clients
                - Make advanced AI technology accessible to all businesses
                
                Why Choose Soft Techniques:
                - Custom AI solutions tailored to your specific needs
                - Experienced team of AI engineers and data scientists
                - Proven track record across multiple industries
                - End-to-end services from strategy to implementation
                
                Ready to transform your business with custom AI? Contact Soft Techniques for a free consultation.
                """,
                "metadata": {"source": "softtechniques_company_overview", "category": "company"}
            },
            {
                "content": """
                Soft Techniques' Comprehensive AI Services
                
                Our Core AI Services:
                - Custom AI Model Development - Tailored AI solutions for your business
                - Machine Learning Solutions - Predictive analytics and data insights
                - Natural Language Processing - Text analysis and language understanding
                - Computer Vision - Image and video analysis systems
                - Intelligent Automation - Process automation and optimization
                - Agentic AI Systems - Autonomous decision-making systems
                
                Industry Expertise:
                - Healthcare - AI-powered patient care and diagnostics
                - Finance - Risk assessment and fraud detection
                - Manufacturing - Quality control and predictive maintenance
                - Retail - Customer analytics and inventory optimization
                - Technology - AI integration and system optimization
                
                Our Approach:
                - Requirements analysis and strategy development
                - Custom model design and training
                - Seamless integration with existing systems
                - Ongoing support and optimization
                
                Interested in our AI services? Schedule a consultation to discuss your specific needs.
                """,
                "metadata": {"source": "softtechniques_services", "category": "services"}
            },
            {
                "content": """
                Soft Techniques' AI Development Process
                
                Our Proven Development Methodology:
                
                Phase 1: Discovery & Strategy
                - Business requirements analysis
                - Data assessment and preparation
                - AI opportunity identification
                - Strategic planning and roadmap
                
                Phase 2: Design & Development
                - Custom AI model architecture
                - Algorithm selection and training
                - System integration planning
                - Security and compliance review
                
                Phase 3: Implementation & Testing
                - Model deployment and integration
                - Comprehensive testing and validation
                - Performance optimization
                - User training and documentation
                
                Phase 4: Support & Optimization
                - Ongoing monitoring and maintenance
                - Performance optimization
                - Continuous learning and improvement
                - Knowledge transfer and support
                
                Our commitment to quality ensures every AI solution delivers measurable business value.
                """,
                "metadata": {"source": "softtechniques_process", "category": "process"}
            },
            {
                "content": """
                Why Choose Soft Techniques for Your AI Needs
                
                Our Competitive Advantages:
                
                Technical Excellence:
                - Latest AI technologies and methodologies
                - Custom solutions, not off-the-shelf products
                - Enterprise-grade security and reliability
                - Scalable architectures that grow with your business
                
                Industry Expertise:
                - Deep understanding of various business domains
                - Proven experience across multiple industries
                - Business-focused approach to AI implementation
                - Measurable ROI and business value delivery
                
                Partnership Approach:
                - Long-term relationships, not just projects
                - Knowledge transfer and team training
                - Ongoing support and optimization
                - Flexible engagement models
                
                Proven Results:
                - 95% client satisfaction rate
                - Average 30% improvement in operational efficiency
                - 40% reduction in manual processes
                - 25% increase in revenue for our clients
                
                Ready to experience the Soft Techniques difference? Contact us for a free consultation.
                """,
                "metadata": {"source": "softtechniques_advantages", "category": "advantages"}
            }
        ]
        
        # Add Soft Techniques documents to knowledge base
        self.add_documents(softtechniques_documents)
        print(f"Added {len(softtechniques_documents)} Soft Techniques company documents to knowledge base")
    
    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Get status information about the knowledge base"""
        return {
            "total_documents": len(self.vectorstore.documents),
            "total_embeddings": len(self.vectorstore.embeddings_cache),
            "persist_directory": self.vectorstore.persist_directory,
            "document_sources": list(set([
                doc.metadata.get("source", "unknown") 
                for doc in self.vectorstore.documents
            ])),
            "document_types": list(set([
                doc.metadata.get("type", "unknown") 
                for doc in self.vectorstore.documents
            ]))
        }

# Initialize simple knowledge base
simple_knowledge_base = SimpleKnowledgeBase()
