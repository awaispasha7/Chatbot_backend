import os
from dotenv import load_dotenv

# Explicitly disable LangSmith to prevent 403 errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Load environment variables
load_dotenv()

class Config:
    # OpenAI Configuration
    # Use dummy key for local testing if not set (will fail on actual API calls but allows server to start)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-dummy-key-for-local-testing-only-not-a-real-key-1234567890abcdef"
    
    # LangSmith Configuration (Explicitly disabled to prevent 403 errors)
    LANGCHAIN_API_KEY = None  # Explicitly set to None
    LANGCHAIN_TRACING_V2 = False  # Explicitly disabled
    LANGCHAIN_PROJECT = None  # Explicitly set to None
    
    # Database Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # N8N Configuration (Optional - will use local processing if not configured)
    N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # Optional
    N8N_API_KEY = os.getenv("N8N_API_KEY")  # Optional
    
    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
