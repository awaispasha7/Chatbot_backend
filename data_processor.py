"""
Local Data Processing Module
Provides data structuring and intermediary calculations without external dependencies
This replaces N8N workflow functionality for cost-effective operation
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

class LocalDataProcessor:
    """
    Local data processor that provides the same functionality as N8N workflows
    but without external dependencies or costs
    """
    
    def __init__(self):
        self.processing_stats = {
            "total_requests": 0,
            "query_types": {},
            "engagement_scores": [],
            "processing_times": []
        }
    
    def process_chat_interaction(self, chat_request: ChatRequest, rag_response: ChatResponse) -> Dict[str, Any]:
        """
        Process chat interaction and return structured data
        This replaces N8N workflow processing
        """
        try:
            self.processing_stats["total_requests"] += 1
            
            # Analyze the query
            query_analysis = self._analyze_query(chat_request.message)
            
            # Analyze the response
            response_analysis = self._analyze_response(rag_response)
            
            # Calculate business insights
            business_insights = self._calculate_business_insights(chat_request, rag_response, query_analysis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(query_analysis, business_insights)
            
            # Update statistics
            self._update_statistics(query_analysis, response_analysis)
            
            structured_data = {
                "query_analysis": query_analysis,
                "response_analysis": response_analysis,
                "business_insights": business_insights,
                "recommendations": recommendations,
                "session_metrics": {
                    "session_id": chat_request.session_id or "default",
                    "timestamp": datetime.now().isoformat(),
                    "user_context": chat_request.context or {}
                },
                "processing_metadata": {
                    "processor": "local",
                    "version": "1.0",
                    "processing_time": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Local data processing completed for query type: {query_analysis['query_type']}")
            return structured_data
            
        except Exception as e:
            logger.error(f"Error in local data processing: {e}")
            return {
                "error": str(e),
                "processing_method": "local",
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query"""
        query_lower = query.lower()
        
        return {
            "query": query,
            "query_length": len(query),
            "word_count": len(query.split()),
            "contains_question": "?" in query,
            "query_type": self._classify_query_type(query),
            "sentiment": self._analyze_sentiment(query),
            "keywords": self._extract_keywords(query),
            "complexity_score": self._calculate_complexity(query)
        }
    
    def _analyze_response(self, rag_response: ChatResponse) -> Dict[str, Any]:
        """Analyze the RAG response"""
        return {
            "response_length": len(rag_response.response),
            "confidence": rag_response.confidence or 0.0,
            "processing_time": rag_response.processing_time or 0.0,
            "source_count": len(rag_response.sources) if rag_response.sources else 0,
            "response_quality": self._assess_response_quality(rag_response),
            "sources_used": [source.get("source", "unknown") for source in (rag_response.sources or [])]
        }
    
    def _calculate_business_insights(self, chat_request: ChatRequest, rag_response: ChatResponse, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business insights from the interaction"""
        engagement_score = self._calculate_engagement_score(chat_request, rag_response, query_analysis)
        
        return {
            "engagement_score": engagement_score,
            "topic_category": self._categorize_topic(chat_request.message),
            "lead_quality": self._assess_lead_quality(chat_request, query_analysis),
            "conversion_potential": self._assess_conversion_potential(chat_request, rag_response),
            "user_intent": self._determine_user_intent(chat_request.message),
            "follow_up_needed": self._needs_follow_up(chat_request, rag_response)
        }
    
    def _generate_recommendations(self, query_analysis: Dict[str, Any], business_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        query_type = query_analysis["query_type"]
        engagement_score = business_insights["engagement_score"]
        
        # Service-related recommendations
        if query_type == "services":
            recommendations.append({
                "action": "schedule_demo",
                "priority": "high",
                "reason": "User interested in services"
            })
            recommendations.append({
                "action": "send_service_brochure",
                "priority": "medium",
                "reason": "Provide detailed service information"
            })
        
        # Pricing-related recommendations
        elif query_type == "pricing":
            recommendations.append({
                "action": "request_quote",
                "priority": "high",
                "reason": "User interested in pricing"
            })
            recommendations.append({
                "action": "contact_sales",
                "priority": "high",
                "reason": "High conversion potential"
            })
        
        # Implementation-related recommendations
        elif query_type == "how_to":
            recommendations.append({
                "action": "download_guide",
                "priority": "medium",
                "reason": "User needs implementation guidance"
            })
            recommendations.append({
                "action": "schedule_consultation",
                "priority": "medium",
                "reason": "Technical consultation needed"
            })
        
        # High engagement recommendations
        if engagement_score > 0.8:
            recommendations.append({
                "action": "priority_follow_up",
                "priority": "high",
                "reason": "High engagement detected"
            })
        
        return recommendations
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain", "meaning"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "process", "steps", "implement", "deploy"]):
            return "how_to"
        elif any(word in query_lower for word in ["why", "benefit", "advantage", "purpose"]):
            return "benefits"
        elif any(word in query_lower for word in ["service", "offer", "provide", "company", "capabilities"]):
            return "services"
        elif any(word in query_lower for word in ["price", "cost", "fee", "rate", "budget"]):
            return "pricing"
        elif any(word in query_lower for word in ["contact", "reach", "speak", "talk"]):
            return "contact"
        else:
            return "general"
    
    def _analyze_sentiment(self, query: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ["good", "great", "excellent", "amazing", "love", "like", "interested"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "problem", "issue"]
        
        query_lower = query.lower()
        
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Return top 10 keywords
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        score = 0.0
        
        # Length factor
        if len(query) > 100:
            score += 0.3
        elif len(query) > 50:
            score += 0.2
        elif len(query) > 20:
            score += 0.1
        
        # Question complexity
        if "?" in query:
            score += 0.2
        
        # Technical terms
        technical_terms = ["api", "integration", "implementation", "deployment", "architecture", "framework"]
        if any(term in query.lower() for term in technical_terms):
            score += 0.3
        
        return min(score, 1.0)
    
    def _assess_response_quality(self, rag_response: ChatResponse) -> str:
        """Assess the quality of the RAG response"""
        if not rag_response.confidence:
            return "unknown"
        
        if rag_response.confidence > 0.8:
            return "high"
        elif rag_response.confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_engagement_score(self, chat_request: ChatRequest, rag_response: ChatResponse, query_analysis: Dict[str, Any]) -> float:
        """Calculate engagement score"""
        score = 0.5  # Base score
        
        # Query length factor
        if query_analysis["query_length"] > 50:
            score += 0.2
        elif query_analysis["query_length"] > 20:
            score += 0.1
        
        # Response confidence factor
        if rag_response.confidence:
            score += rag_response.confidence * 0.3
        
        # Source count factor
        if rag_response.sources:
            score += min(len(rag_response.sources) * 0.1, 0.2)
        
        # Complexity factor
        score += query_analysis["complexity_score"] * 0.2
        
        return min(score, 1.0)
    
    def _categorize_topic(self, query: str) -> str:
        """Categorize the topic of the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["agentic", "ai", "artificial intelligence", "machine learning"]):
            return "agentic_ai"
        elif any(word in query_lower for word in ["service", "offer", "provide", "capability"]):
            return "services"
        elif any(word in query_lower for word in ["implement", "deploy", "setup", "integration"]):
            return "implementation"
        elif any(word in query_lower for word in ["cost", "price", "budget", "pricing"]):
            return "pricing"
        elif any(word in query_lower for word in ["contact", "reach", "speak", "demo"]):
            return "contact"
        else:
            return "general"
    
    def _assess_lead_quality(self, chat_request: ChatRequest, query_analysis: Dict[str, Any]) -> str:
        """Assess lead quality based on query"""
        query_type = query_analysis["query_type"]
        complexity = query_analysis["complexity_score"]
        
        if query_type in ["pricing", "services"] and complexity > 0.5:
            return "high"
        elif query_type in ["how_to", "benefits"] and complexity > 0.3:
            return "medium"
        else:
            return "low"
    
    def _assess_conversion_potential(self, chat_request: ChatRequest, rag_response: ChatResponse) -> str:
        """Assess conversion potential"""
        if not rag_response.confidence:
            return "unknown"
        
        if rag_response.confidence > 0.8 and len(chat_request.message) > 30:
            return "high"
        elif rag_response.confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _determine_user_intent(self, query: str) -> str:
        """Determine user intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["buy", "purchase", "order", "get"]):
            return "purchase"
        elif any(word in query_lower for word in ["learn", "understand", "know", "explain"]):
            return "learn"
        elif any(word in query_lower for word in ["help", "support", "problem", "issue"]):
            return "support"
        elif any(word in query_lower for word in ["demo", "trial", "test", "try"]):
            return "demo"
        else:
            return "explore"
    
    def _needs_follow_up(self, chat_request: ChatRequest, rag_response: ChatResponse) -> bool:
        """Determine if follow-up is needed"""
        query_type = self._classify_query_type(chat_request.message)
        
        # High-priority follow-up cases
        if query_type in ["pricing", "services"]:
            return True
        
        # Medium-priority follow-up cases
        if query_type in ["how_to", "contact"]:
            return True
        
        # Low-priority follow-up cases
        if query_type in ["definition", "benefits"]:
            return False
        
        return False
    
    def _update_statistics(self, query_analysis: Dict[str, Any], response_analysis: Dict[str, Any]):
        """Update processing statistics"""
        query_type = query_analysis["query_type"]
        
        # Update query type counts
        if query_type in self.processing_stats["query_types"]:
            self.processing_stats["query_types"][query_type] += 1
        else:
            self.processing_stats["query_types"][query_type] = 1
        
        # Update engagement scores
        self.processing_stats["engagement_scores"].append(response_analysis.get("confidence", 0.0))
        
        # Update processing times
        if response_analysis.get("processing_time"):
            self.processing_stats["processing_times"].append(response_analysis["processing_time"])
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_engagement = sum(self.processing_stats["engagement_scores"]) / len(self.processing_stats["engagement_scores"]) if self.processing_stats["engagement_scores"] else 0
        avg_processing_time = sum(self.processing_stats["processing_times"]) / len(self.processing_stats["processing_times"]) if self.processing_stats["processing_times"] else 0
        
        return {
            "total_requests": self.processing_stats["total_requests"],
            "query_type_distribution": self.processing_stats["query_types"],
            "average_engagement_score": avg_engagement,
            "average_processing_time": avg_processing_time,
            "most_common_query_type": max(self.processing_stats["query_types"], key=self.processing_stats["query_types"].get) if self.processing_stats["query_types"] else "none"
        }

# Initialize local data processor
local_data_processor = LocalDataProcessor()
