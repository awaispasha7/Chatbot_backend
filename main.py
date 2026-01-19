# Explicitly disable LangSmith to prevent 403 errors
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional
import uuid
import time
from datetime import datetime

from models import ChatRequest, ChatResponse, ChatMessage
from rag_system import rag_system
from n8n_integration import n8n_integration
from knowledge_base import knowledge_base
from scheduling_system import consultation_scheduler
from consultation_logger import consultation_logger
from config import Config

# Football Analysis imports - wrapped in try/except to prevent startup failures
FOOTBALL_ANALYSIS_AVAILABLE = False
football_videos = None
football_jobs = None
football_results = None
VideoProcessor = None
get_football_settings = None
start_cleanup_scheduler = None
check_ffmpeg_installed = lambda: False

try:
    print("=" * 60)
    print("ATTEMPTING TO LOAD FOOTBALL ANALYSIS MODULES...")
    print("=" * 60)
    
    # Try importing each module individually to see which one fails
    try:
        from app.routers import videos as football_videos
        print("✅ Successfully imported app.routers.videos")
    except ImportError as e:
        print(f"❌ Failed to import app.routers.videos: {e}")
        raise
    
    try:
        from app.routers import jobs as football_jobs
        print("✅ Successfully imported app.routers.jobs")
    except ImportError as e:
        print(f"❌ Failed to import app.routers.jobs: {e}")
        raise
    
    try:
        from app.routers import results as football_results
        print("✅ Successfully imported app.routers.results")
    except ImportError as e:
        print(f"❌ Failed to import app.routers.results: {e}")
        raise
    
    try:
        from app.config import get_settings as get_football_settings
        print("✅ Successfully imported app.config.get_settings")
    except ImportError as e:
        print(f"❌ Failed to import app.config.get_settings: {e}")
        raise
    
    try:
        from app.services.video_processor import VideoProcessor
        print("✅ Successfully imported app.services.video_processor.VideoProcessor")
    except ImportError as e:
        print(f"❌ Failed to import VideoProcessor: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    try:
        from app.services.file_cleanup import start_cleanup_scheduler
        print("✅ Successfully imported app.services.file_cleanup.start_cleanup_scheduler")
    except ImportError as e:
        print(f"❌ Failed to import start_cleanup_scheduler: {e}")
        raise
    
    try:
        from app.utils.ffprobe import check_ffmpeg_installed
        print("✅ Successfully imported app.utils.ffprobe.check_ffmpeg_installed")
    except ImportError as e:
        print(f"❌ Failed to import check_ffmpeg_installed: {e}")
        raise
    
    FOOTBALL_ANALYSIS_AVAILABLE = True
    print("=" * 60)
    print("✅ ALL FOOTBALL ANALYSIS MODULES LOADED SUCCESSFULLY")
    print("=" * 60)
    
except ImportError as e:
    print("=" * 60)
    print(f"❌ FOOTBALL ANALYSIS MODULES FAILED TO LOAD")
    print(f"Error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    print("=" * 60)
    print("⚠️ Football Analysis endpoints will not be available.")
    print("This usually means dependencies are missing:")
    print("  - ultralytics>=8.0.0")
    print("  - opencv-python-headless>=4.8.0")
    print("  - yt-dlp>=2023.12.0")
    print("  - deep-sort-realtime>=1.3.2")
    print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI Chatbot API",
    description="A RAG-based chatbot for agentic AI services company",
    version="1.0.0"
)

# Configure CORS from environment
# ALLOWED_ORIGINS can be a comma-separated list of origins, e.g.
# "https://softtechniques.com, http://localhost:3000"
# Default includes your custom domain and localhost for development
_default_origins = "https://softtechniques.com,http://localhost:3000,http://127.0.0.1:5500,http://localhost:5500"
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", _default_origins)
_allowed_origins_list = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

_use_wildcard = len(_allowed_origins_list) == 1 and _allowed_origins_list[0] == "*"
_allow_credentials = not _use_wildcard  # Starlette forbids credentials with wildcard

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for admin dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include Football Analysis routers (if available)
print("=" * 60)
print("REGISTERING ROUTES")
print("=" * 60)
print(f"FOOTBALL_ANALYSIS_AVAILABLE: {FOOTBALL_ANALYSIS_AVAILABLE}")
print(f"football_videos: {football_videos is not None}")
print(f"football_jobs: {football_jobs is not None}")
print(f"football_results: {football_results is not None}")
print("=" * 60)

if FOOTBALL_ANALYSIS_AVAILABLE and football_videos and football_jobs and football_results:
    app.include_router(football_videos.router)
    app.include_router(football_jobs.router)
    app.include_router(football_results.router)
    print("✅ Football Analysis routes registered successfully")
    if hasattr(football_videos.router, 'prefix'):
        print(f"  - {football_videos.router.prefix}/upload")
        print(f"  - {football_videos.router.prefix}/youtube")
    if hasattr(football_jobs.router, 'prefix'):
        print(f"  - {football_jobs.router.prefix}/{{jobId}}")
    if hasattr(football_results.router, 'prefix'):
        print(f"  - {football_results.router.prefix}/{{jobId}}.mp4")
else:
    print("⚠️ Football Analysis routes not available - dependencies may be missing")
    print(f"  FOOTBALL_ANALYSIS_AVAILABLE: {FOOTBALL_ANALYSIS_AVAILABLE}")
    print(f"  football_videos: {football_videos}")
    print(f"  football_jobs: {football_jobs}")
    print(f"  football_results: {football_results}")
print("=" * 60)

# In-memory session storage (use Redis or database in production)
sessions: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session"""
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(),
            "messages": [],
            "context": {}
        }
    return sessions[session_id]

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Agentic AI Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs",
            "dashboard": "/dashboard"
        }
    }

@app.get("/dashboard")
async def dashboard():
    """Serve the admin dashboard"""
    from fastapi.responses import FileResponse
    import os
    dashboard_path = os.path.join("static", "admin_dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "rag_system": "operational",
            "knowledge_base": "operational",
            "n8n_integration": "enabled" if Config.N8N_WEBHOOK_URL else "disabled (using local processing)"
        }
    }
    
    # Add football analysis status if available
    if FOOTBALL_ANALYSIS_AVAILABLE and VideoProcessor:
        health_data["services"]["football_analysis"] = "operational"
        health_data["services"]["model_loaded"] = VideoProcessor._model is not None
        health_data["services"]["ffmpeg_available"] = check_ffmpeg_installed()
    else:
        health_data["services"]["football_analysis"] = "unavailable (dependencies missing)"
    
    return health_data

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Main chat endpoint that processes user messages and returns AI responses
    """
    try:
        # Generate session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Get or create session
        session = get_session(session_id)
        
        # Add user message to session
        user_message = ChatMessage(
            role="user",
            content=chat_request.message,
            timestamp=datetime.now()
        )
        session["messages"].append(user_message.dict())
        
        # Process with RAG system
        rag_response = rag_system.chat(
            query=chat_request.message,
            session_id=session_id,
            user_context=chat_request.context or session.get("context", {})
        )
        
        # Send to n8n workflow for data structuring (optional component)
        n8n_result = n8n_integration.send_to_n8n_workflow(chat_request, rag_response)
        
        # Process structured data from n8n if available
        if n8n_result:
            structured_data = n8n_integration.process_structured_data(n8n_result)
            # Update session context with structured data
            session["context"].update(structured_data)
        
        # Add assistant message to session
        assistant_message = ChatMessage(
            role="assistant",
            content=rag_response.response,
            timestamp=datetime.now()
        )
        session["messages"].append(assistant_message.dict())
        
        # Update session
        sessions[session_id] = session
        
        return rag_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Get chat history for a specific session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Get conversation summary from RAG system
    conversation_summary = rag_system.get_conversation_summary(session_id)
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "message_count": len(session["messages"]),
        "messages": session["messages"],
        "context": session["context"],
        "conversation_summary": conversation_summary
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear conversation memory
    rag_system.clear_conversation_memory(session_id)
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "created_at": session["created_at"],
                "message_count": len(session["messages"])
            }
            for session_id, session in sessions.items()
        ]
    }

@app.post("/knowledge-base/add-text")
async def add_text_to_knowledge_base(texts: list[str], metadata: Optional[list[dict]] = None):
    """Add text documents to the knowledge base"""
    try:
        knowledge_base.add_documents_from_text(texts, metadata)
        return {"message": f"Added {len(texts)} documents to knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@app.post("/knowledge-base/upload-pdf")
async def upload_pdf_to_knowledge_base(file: UploadFile = File(...)):
    """Upload and process a PDF file to add to the knowledge base"""
    try:
        # Check if file is PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add to knowledge base
        knowledge_base.add_documents_from_file(file_path)
        
        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "filename": file.filename,
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/knowledge-base/search")
async def search_knowledge_base(query: str, k: int = 5):
    """Search the knowledge base"""
    try:
        results = knowledge_base.search(query, k=k)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {str(e)}")

@app.get("/knowledge-base/status")
async def get_knowledge_base_status():
    """Get knowledge base status and statistics"""
    try:
        status = knowledge_base.get_knowledge_base_status()
        return {
            "status": "success",
            "knowledge_base": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge base status: {str(e)}")

@app.post("/knowledge-base/initialize")
async def initialize_knowledge_base():
    """Initialize the knowledge base with default agentic AI content"""
    try:
        knowledge_base.initialize_with_agentic_ai_content()
        return {"message": "Knowledge base initialized with agentic AI content"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing knowledge base: {str(e)}")

@app.get("/n8n/status")
async def get_n8n_status():
    """Get N8N integration status"""
    return {
        "configured": bool(Config.N8N_WEBHOOK_URL),
        "webhook_url": Config.N8N_WEBHOOK_URL,
        "api_key_configured": bool(Config.N8N_API_KEY),
        "fallback_mode": "local_data_processing" if not Config.N8N_WEBHOOK_URL else "n8n_workflow"
    }

@app.get("/analytics/processing-stats")
async def get_processing_statistics():
    """Get processing statistics from local data processor"""
    try:
        from data_processor import local_data_processor
        stats = local_data_processor.get_processing_statistics()
        return {
            "status": "success",
            "statistics": stats,
            "processing_method": "local_data_processing"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "processing_method": "local_data_processing"
        }

@app.get("/memory/sessions")
async def get_all_memory_sessions():
    """Get all active conversation memory sessions"""
    try:
        sessions = rag_system.get_all_sessions()
        return {
            "active_sessions": sessions,
            "total_sessions": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory sessions: {str(e)}")

@app.get("/memory/sessions/{session_id}")
async def get_conversation_memory(session_id: str):
    """Get conversation memory for a specific session"""
    try:
        summary = rag_system.get_conversation_summary(session_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation memory: {str(e)}")

@app.delete("/memory/sessions/{session_id}")
async def clear_conversation_memory(session_id: str):
    """Clear conversation memory for a specific session"""
    try:
        success = rag_system.clear_conversation_memory(session_id)
        if success:
            return {"message": f"Conversation memory cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation memory: {str(e)}")








# Consultation Scheduling Endpoints
@app.get("/consultation/available-slots")
async def get_available_consultation_slots():
    """Get available consultation time slots"""
    try:
        slots = consultation_scheduler.get_available_slots()
        return {
            "status": "success",
            "available_slots": slots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available slots: {str(e)}")

@app.post("/consultation/schedule")
async def schedule_consultation(
    request: Request,
    name: str = "",
    email: str = "",
    phone: str = "",
    company: str = "",
    preferred_date: str = "",
    preferred_time: str = "",
    message: str = ""
):
    """Schedule a new consultation"""
    try:
        # Attempt to parse JSON body (for programmatic scheduling)
        try:
            body = await request.json()
            if isinstance(body, dict):
                name = body.get("name", name)
                email = body.get("email", email)
                phone = body.get("phone", phone)
                company = body.get("company", company)
                preferred_date = body.get("preferred_date", preferred_date)
                preferred_time = body.get("preferred_time", preferred_time)
                message = body.get("message", message)
        except Exception:
            # If no JSON body, continue with query/form params
            pass

        # Get client information for logging
        ip_address = request.client.host if request else ""
        user_agent = request.headers.get("user-agent", "") if request else ""
        
        result = consultation_scheduler.schedule_consultation(
            name=name,
            email=email,
            phone=phone,
            company=company,
            preferred_date=preferred_date,
            preferred_time=preferred_time,
            message=message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling consultation: {str(e)}")

@app.get("/consultation/available-slots")
async def get_available_consultation_slots():
    """Get available consultation time slots"""
    try:
        slots = consultation_scheduler.get_available_slots()
        return {
            "status": "success",
            "available_slots": slots
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available slots: {str(e)}")

@app.get("/consultation/status/{consultation_id}")
async def get_consultation_status(consultation_id: str):
    """Get status of a consultation request"""
    try:
        result = consultation_scheduler.get_consultation_status(consultation_id)
        if result["found"]:
            return result
        else:
            raise HTTPException(status_code=404, detail="Consultation request not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting consultation status: {str(e)}")

@app.get("/consultation/all")
async def get_all_consultations():
    """Get all consultation requests (admin endpoint)"""
    try:
        requests = consultation_scheduler.get_all_requests()
        return {
            "status": "success",
            "total_requests": len(requests),
            "requests": requests
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting consultations: {str(e)}")

@app.put("/consultation/update-status/{consultation_id}")
async def update_consultation_status(consultation_id: str, status: str):
    """Update consultation status (admin endpoint)"""
    try:
        result = consultation_scheduler.update_consultation_status(consultation_id, status)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail="Consultation request not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating consultation status: {str(e)}")

@app.delete("/consultation/delete/{consultation_id}")
async def delete_consultation(consultation_id: str):
    """Delete a consultation request (admin endpoint)"""
    try:
        result = consultation_scheduler.delete_consultation(consultation_id)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail="Consultation request not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting consultation: {str(e)}")

# Admin Logging and Team Management Endpoints
@app.get("/admin/logs/recent")
async def get_recent_consultation_logs(hours: int = 24):
    """Get recent consultation logs (admin endpoint)"""
    try:
        logs = consultation_logger.get_recent_logs(hours=hours)
        return {
            "status": "success",
            "total_logs": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent logs: {str(e)}")

@app.get("/admin/logs/status/{status}")
async def get_logs_by_status(status: str):
    """Get logs by status (admin endpoint)"""
    try:
        logs = consultation_logger.get_logs_by_status(status)
        return {
            "status": "success",
            "total_logs": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs by status: {str(e)}")

@app.get("/admin/logs/date-range")
async def get_logs_by_date_range(start_date: str, end_date: str):
    """Get logs by date range (admin endpoint)"""
    try:
        logs = consultation_logger.get_logs_by_date_range(start_date, end_date)
        return {
            "status": "success",
            "total_logs": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs by date range: {str(e)}")

@app.get("/admin/stats")
async def get_consultation_stats():
    """Get consultation statistics (admin endpoint)"""
    try:
        stats = consultation_logger.get_consultation_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/admin/team")
async def get_team_members():
    """Get all team members (admin endpoint)"""
    try:
        team_members = consultation_logger.get_team_members()
        return {
            "status": "success",
            "total_members": len(team_members),
            "team_members": team_members
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting team members: {str(e)}")

@app.post("/admin/team/add")
async def add_team_member(name: str, email: str, role: str, phone: str = ""):
    """Add a new team member (admin endpoint)"""
    try:
        result = consultation_logger.add_team_member(name, email, role, phone)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding team member: {str(e)}")

@app.delete("/admin/team/remove/{email}")
async def remove_team_member(email: str):
    """Remove a team member (admin endpoint)"""
    try:
        result = consultation_logger.remove_team_member(email)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail="Team member not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing team member: {str(e)}")

@app.post("/admin/clear-all-logs")
async def clear_all_consultation_logs():
    """Clear all consultation logs (admin endpoint)"""
    try:
        result = consultation_logger.clear_all_logs()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize football analysis components on startup"""
    if not FOOTBALL_ANALYSIS_AVAILABLE:
        print("⚠️ Football Analysis not available - skipping initialization")
        return
    
    try:
        import asyncio
        football_settings = get_football_settings()
        football_settings.ensure_directories()
        
        # Check ffmpeg
        if not check_ffmpeg_installed():
            print("WARNING: FFmpeg/FFprobe not found in PATH. Video processing will fail.")
        
        # Pre-load YOLO model
        print("Loading YOLO model for football analysis...")
        if VideoProcessor.load_model():
            print(f"Model loaded. Classes: {VideoProcessor.get_class_names()}")
        else:
            print("WARNING: Failed to load model. Processing will fail.")
        
        # Start cleanup scheduler
        asyncio.create_task(start_cleanup_scheduler())
        print("Football analysis backend initialized successfully.")
    except Exception as e:
        print(f"WARNING: Error initializing football analysis backend: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the entire app if football analysis fails to initialize

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )
