import requests
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from config import Config
from models import N8NWebhookPayload, ChatRequest, ChatResponse
from data_processor import local_data_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class N8NIntegration:
    def __init__(self):
        self.webhook_url = Config.N8N_WEBHOOK_URL
        self.api_key = Config.N8N_API_KEY
        self.enabled = bool(self.webhook_url)  # Only enable if webhook URL is provided
        
        if self.enabled:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" if self.api_key else None
            }
            # Remove None values from headers
            self.headers = {k: v for k, v in self.headers.items() if v is not None}
            logger.info("N8N integration enabled")
        else:
            self.headers = {}
            logger.info("N8N integration disabled - no webhook URL provided")
    
    def send_to_n8n_workflow(self, chat_request: ChatRequest, rag_response: ChatResponse) -> Optional[Dict[str, Any]]:
        """
        Send data to n8n workflow for processing and structuring
        This is an optional component for data structuring and intermediary calculations
        """
        if not self.enabled:
            logger.info("N8N integration disabled, using local data processing")
            return local_data_processor.process_chat_interaction(chat_request, rag_response)
        
        try:
            # Prepare payload for n8n workflow
            payload = N8NWebhookPayload(
                query=chat_request.message,
                session_id=chat_request.session_id or "default",
                user_context=chat_request.context,
                timestamp=datetime.now()
            )
            
            # Add RAG response data
            n8n_data = {
                "input": payload.dict(),
                "rag_response": {
                    "response": rag_response.response,
                    "sources": rag_response.sources,
                    "confidence": rag_response.confidence,
                    "processing_time": rag_response.processing_time
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "workflow_type": "chatbot_data_processing"
                }
            }
            
            # Send to n8n workflow
            response = requests.post(
                self.webhook_url,
                json=n8n_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Successfully sent data to n8n workflow: {result}")
                return result
            else:
                print(f"N8N workflow returned status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to n8n workflow: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in n8n integration: {e}")
            return local_data_processor.process_chat_interaction(chat_request, rag_response)
    
    
    def process_structured_data(self, n8n_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the structured data returned from n8n workflow
        This could include calculations, data enrichment, or formatting
        """
        if not n8n_response:
            return {}
        
        try:
            # Extract processed data from n8n response
            processed_data = n8n_response.get("processed_data", {})
            
            # Perform additional processing if needed
            enhanced_data = {
                "original_query": processed_data.get("query", ""),
                "structured_response": processed_data.get("structured_response", ""),
                "calculations": processed_data.get("calculations", {}),
                "enrichment": processed_data.get("enrichment", {}),
                "recommendations": processed_data.get("recommendations", []),
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "n8n_workflow_id": n8n_response.get("workflow_id"),
                    "processing_time": n8n_response.get("processing_time")
                }
            }
            
            return enhanced_data
            
        except Exception as e:
            print(f"Error processing structured data: {e}")
            return {}
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific n8n workflow"""
        if not self.api_key or not workflow_id:
            return None
        
        try:
            # This would require n8n API endpoint for workflow status
            # Implementation depends on your n8n setup
            status_url = f"{self.webhook_url.replace('/webhook', '')}/api/v1/workflows/{workflow_id}/status"
            
            response = requests.get(
                status_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get workflow status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting workflow status: {e}")
            return None

# Initialize N8N integration
n8n_integration = N8NIntegration()
