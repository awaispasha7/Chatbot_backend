# Agentic AI Chatbot

A sophisticated RAG-based chatbot designed for agentic AI services companies. This chatbot provides intelligent responses about agentic AI concepts, applications, and implementation strategies using OpenAI's GPT models and LangChain framework.

## Features

- **RAG (Retrieval-Augmented Generation)**: Combines knowledge retrieval with AI generation for accurate responses
- **LangChain Integration**: Built with LangChain for robust AI workflows
- **N8N Workflow Integration**: Optional data structuring and intermediary calculations via N8N workflows (with local fallback)
- **FastAPI Backend**: High-performance API with automatic documentation
- **Modern Web Interface**: Beautiful, responsive chat interface
- **Widget Integration**: Easy-to-embed chatbot widget for any website
- **Session Management**: Persistent conversation sessions
- **Knowledge Base**: Vector-based document storage with ChromaDB

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI API   │    │   RAG System    │
│                 │◄──►│                 │◄──►│                 │
│  - Chat UI      │    │  - Chat Endpoint│    │  - LangChain    │
│  - Widget       │    │  - Session Mgmt │    │  - OpenAI GPT   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   N8N Workflow  │
                       │                 │
                       │  - Data Struct. │
                       │  - Calculations │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Knowledge Base  │
                       │                 │
                       │  - ChromaDB     │
                       │  - Embeddings   │
                       └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd Chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional
N8N_WEBHOOK_URL=your_n8n_webhook_url_here      # Optional
```

### 3. Initialize the System

```bash
# Initialize the knowledge base and test the system
python initialize.py
```

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py
```

The chatbot will be available at:
- **Web Interface**: http://localhost:8000/chat-interface
- **API Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000/chat

## API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is agentic AI?",
       "session_id": "optional_session_id"
     }'
```

### Response Format

```json
{
  "response": "Agentic AI refers to artificial intelligence systems...",
  "session_id": "session_123",
  "sources": [
    {
      "content": "Agentic AI refers to artificial intelligence systems...",
      "source": "agentic_ai_overview",
      "score": 0.95
    }
  ],
  "confidence": 0.8,
  "processing_time": 1.2
}
```

## Website Integration

### Option 1: Widget Integration (Recommended)

Add this script to your website's HTML:

```html
<!-- Note: Static HTML files have been removed. Use the API endpoints directly. -->
<script>
  // Use the API endpoint directly
  const apiUrl = "http://localhost:8000/chat";
  // Implement your own chat interface using the API
</script>
```

### Option 2: Custom Integration

```javascript
// Initialize the chatbot widget
const chatbot = new AgenticAIChatbotWidget({
  apiUrl: 'http://localhost:8000/chat',
  position: 'bottom-right',
  primaryColor: '#4f46e5',
  secondaryColor: '#7c3aed',
  title: 'Agentic AI Assistant',
  subtitle: 'Ask me about agentic AI'
});

// Control the chatbot programmatically
chatbot.openChat();
chatbot.closeChat();
chatbot.sendMessage('Hello!');
```

### Option 3: Iframe Integration

```html
<iframe 
  src="http://localhost:8000/chat-interface" 
  width="400" 
  height="600"
  frameborder="0">
</iframe>
```

## N8N Workflow Integration (Optional)

The chatbot includes optional N8N workflow integration for data structuring and intermediary calculations. If not configured, it will use local data processing instead. Configure your N8N webhook URL in the `.env` file:

```env
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/chatbot
N8N_API_KEY=your_n8n_api_key
```

### N8N Workflow Payload

The chatbot sends the following data to your N8N workflow:

```json
{
  "input": {
    "query": "What is agentic AI?",
    "session_id": "session_123",
    "user_context": {},
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "rag_response": {
    "response": "Agentic AI refers to...",
    "sources": [...],
    "confidence": 0.8,
    "processing_time": 1.2
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "workflow_type": "chatbot_data_processing"
  }
}
```

## Knowledge Base Management

### Adding Documents

```python
from knowledge_base import knowledge_base

# Add text documents
texts = ["Your document content here..."]
metadata = [{"source": "custom_doc", "category": "custom"}]
knowledge_base.add_documents_from_text(texts, metadata)

# Add from file
knowledge_base.add_documents_from_file("path/to/document.pdf")
```

### Searching the Knowledge Base

```python
# Search for relevant documents
results = knowledge_base.search("agentic AI applications", k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `LANGCHAIN_API_KEY` | LangSmith API key (optional) | - |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | false |
| `LANGCHAIN_PROJECT` | LangSmith project name | agentic-ai-chatbot |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage directory | ./chroma_db |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `N8N_WEBHOOK_URL` | N8N workflow webhook URL (optional) | - |
| `N8N_API_KEY` | N8N API key (optional) | - |

### Model Configuration

```python
# In config.py
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
CHAT_MODEL = "gpt-4-turbo-preview"          # OpenAI chat model
TEMPERATURE = 0.7                           # Response creativity
MAX_TOKENS = 1000                           # Maximum response length
CHUNK_SIZE = 1000                           # Document chunk size
CHUNK_OVERLAP = 200                         # Chunk overlap
TOP_K_RESULTS = 5                           # Number of search results
```

## Development

### Project Structure

```
Chatbot/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── models.py              # Pydantic models
├── rag_system.py          # RAG implementation
├── knowledge_base.py      # Knowledge base management
├── n8n_integration.py     # N8N workflow integration
├── initialize.py          # System initialization
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
└── README.md             # This file
```

### Running in Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Test the knowledge base
python -c "from knowledge_base import knowledge_base; print(knowledge_base.search('agentic AI', k=3))"

# Test the API
curl -X GET "http://localhost:8000/health"
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup

1. Set up a production environment with proper API keys
2. Configure N8N workflow endpoints
3. Set up monitoring and logging
4. Configure CORS for your domain
5. Set up SSL/TLS certificates

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Ensure your API key is correctly set in the `.env` file
2. **ChromaDB Issues**: Delete the `chroma_db` directory and reinitialize
3. **N8N Connection Issues**: Check your webhook URL and API key
4. **Memory Issues**: Reduce `CHUNK_SIZE` and `TOP_K_RESULTS` in config

### Logs

Check the console output for detailed error messages. For production, consider setting up proper logging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Built with ❤️ for the agentic AI community**
