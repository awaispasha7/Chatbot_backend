from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: Optional[List[Dict[str, str]]] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_id: str

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

class N8NWebhookPayload(BaseModel):
    query: str
    session_id: str
    user_context: Optional[Dict[str, Any]] = None
    timestamp: datetime
