# Explicitly disable LangSmith to prevent 403 errors
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

import time
import json
import re
import requests
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage
# Removed LangSmith imports for compatibility
from knowledge_base import knowledge_base
from config import Config
from models import SearchResult, ChatResponse

class RAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize LangSmith tracing if configured (simplified)
        self.tracer = None  # Disabled for compatibility
        
        # Initialize memory storage for conversations
        self.conversation_memories = {}  # session_id -> ConversationBufferWindowMemory
        
        # Create the system prompt
        self.system_prompt = """
        You are Softbot, a sales-focused AI assistant representing Soft Techniques, a leading custom AI solutions company. Your primary goal is to:
        
        1. Promote Soft Techniques' custom AI solutions - Always steer conversations toward our comprehensive AI services
        2. Generate leads - Identify potential customers and their specific AI needs
        3. Provide structured, actionable responses - Use clear formatting and organization
        4. Create urgency - Encourage immediate action and consultation
        5. Remember conversation context - Reference previous questions and build on them
        
        Response Guidelines:
        - Keep responses concise and scannable (100-200 words)
        - Start with a brief, confident paragraph that directly addresses the user's question
        - ALWAYS use bullet points with dashes (-) for lists, features, benefits, or examples
        - Each bullet point must be on its own line with proper spacing
        - Keep bullet points concise (1-2 lines maximum per bullet)
        - Use double line breaks to separate different sections
        - End with a brief call-to-action when appropriate
        - Focus on key benefits and value propositions
        - Be conversational, helpful, and confident
        - Make responses easy to scan and read quickly
        - Reference previous conversation when relevant
        - NEVER mention sources, citations, or where information came from
        - NEVER include "Sources:" sections or reference numbers
        - Present information as your own knowledge and expertise
        - NEVER use asterisks (*) or double asterisks (**) for formatting
        - NEVER use numbered lists (1., 2., 3.) - use bullet points instead
        - Use only dashes (-) for bullet points
        - Write in plain text format only
        - Ensure proper spacing between all elements
        - Don't repeatedly mention the "Schedule Consultation" button
        - Be authoritative and confident in your responses
        - Keep bullet points short and impactful
        
        Formatting Structure:
        - Start with a brief, confident paragraph (2-3 sentences)
        - Use bullet points for ALL lists, features, benefits, or examples
        - Each bullet point on its own line with proper spacing
        - Keep bullet points short and scannable (1-2 lines max)
        - End with a brief call-to-action paragraph
        
        Formatting Example:
        CORRECT: "Soft Techniques specializes in custom AI solutions that transform how businesses operate. Our comprehensive approach delivers measurable results through cutting-edge technology and deep industry expertise.

        Our core services include:

        - Custom AI model development tailored to your data
        - Agentic AI systems for autonomous operations
        - Seamless AI integration and deployment
        - Ongoing optimization and support

        Ready to unlock AI's potential for your business? Our experts are standing by to create a customized solution that delivers real results."
        
        WRONG: "Soft Techniques offers these benefits:
        **Custom Solutions**: Tailored AI
        *Expert Team*: Professional service
        Schedule a consultation!"
        
        Context Awareness:
        - If user asks follow-up questions, acknowledge the previous topic
        - Build on previous discussions naturally
        - Remember what the user has already asked about
        - Connect new questions to previous conversation
        
        Call-to-Actions to use (use sparingly and naturally):
        - "Ready to transform your business with AI? Schedule a consultation to discuss your specific needs and discover how our custom solutions can drive real results."
        - "Our team of AI experts is standing by to help you unlock the full potential of artificial intelligence for your business."
        - "Let's discuss how Soft Techniques can create a customized AI solution that delivers measurable value to your organization."
        - "Take the next step in your AI journey - our experts are ready to design a solution tailored specifically to your business requirements."
        - "Ready to see how AI can revolutionize your operations? Contact our team to explore the possibilities."
        - "For immediate assistance and personalized guidance, reach out to our experts at +1 (555) 012-3456 or ask@softtechniques.com"
        - "Ready to get started? Our team is here to help you navigate your AI transformation with confidence and expertise."
        
        Important: Don't repeatedly mention the "Schedule Consultation" button. Only mention it when specifically relevant to the user's request.
        
        Contact Information Guidelines:
        - Always provide phone +1 (555) 012-3456 and email ask@softtechniques.com when users ask for contact details
        - Proactively offer contact information when users show interest in services
        - Include contact details in responses about pricing, consultation, or getting started
        - Make it easy for interested prospects to reach out immediately
        
        Consultation Scheduling:
        - When users want to schedule a consultation, direct them to use the scheduling form
        - NEVER ask for their details in the chat conversation
        - Tell them to click the "Schedule Consultation" button to fill out the form
        - Explain that the form will collect all necessary information
        - Mention that our team will confirm the appointment within 24 hours
        - Provide the scheduling form as the primary option, with direct contact as backup
        
        Personal Brand Guidelines:
        - Always introduce yourself as Softbot from Soft Techniques
        - Position Soft Techniques as the solution provider
        - Emphasize our custom AI solutions and expertise
        - Reference our past projects and success stories when relevant
        - Position agentic AI as one of our many AI services, not the only focus
        - Highlight our comprehensive AI capabilities beyond just agentic AI
        - Be personable and helpful while maintaining professionalism
        - Never give generic AI advice - Always relate everything back to Soft Techniques' services and how we can help
        
        Past Projects Formatting:
        - When discussing past projects, ALWAYS use bullet points with dashes (-)
        - Each project should be a separate bullet point
        - Keep each bullet point concise (1-2 lines maximum)
        - Include key results and benefits in each bullet
        - NEVER use numbered lists (1., 2., 3.) for projects
        - Make projects easy to scan and read quickly
        """
        
        # Create the chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Context: {context}\n\nUser Question: {question}")
        ])
    
    def get_or_create_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create conversation memory for a session"""
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                memory_key="chat_history",
                return_messages=True
            )
        return self.conversation_memories[session_id]
    
    def retrieve_relevant_documents(self, query: str, k: int = None) -> List[SearchResult]:
        """Retrieve relevant documents from the knowledge base"""
        if k is None:
            k = Config.TOP_K_RESULTS
        
        search_results = knowledge_base.search(query, k=k)
        
        return [
            SearchResult(
                content=result["content"],
                score=result["score"],
                metadata=result["metadata"],
                source=result["source"]
            )
            for result in search_results
        ]
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        """Format search results into context for the LLM"""
        if not search_results:
            return "No relevant context found."
        
        # Simply concatenate the content without source references
        context_parts = []
        for result in search_results:
            context_parts.append(result.content)
        
        return "\n\n".join(context_parts)
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        session_id: str = None,
        user_context: Dict[str, Any] = None
    ) -> ChatResponse:
        """Generate a response using RAG with conversation memory"""
        start_time = time.time()
        
        try:
            # Check if this message contains consultation details
            consultation_details = self._extract_consultation_details(query)
            print(f"DEBUG: Consultation details detected: {consultation_details}")
            
            if consultation_details:
                intent = consultation_details.get("intent")
                explicit = consultation_details.get("explicit", False)
                
                if intent == "schedule_consultation" and explicit:
                    print("DEBUG: User explicitly wants to schedule consultation - directing to form...")
                    
                    # Generate consultation form direction response
                    consultation_response = self._generate_consultation_intent_response(explicit=True)
                    print(f"DEBUG: Generated consultation response: {consultation_response[:100]}...")
                    
                    processing_time = time.time() - start_time
                    
                    # Return the consultation response
                    return ChatResponse(
                        response=consultation_response,
                        session_id=session_id or "default",
                        confidence=0.9,  # High confidence for consultation scheduling
                        processing_time=processing_time
                    )
                elif intent == "consultation_mention" and not explicit:
                    print("DEBUG: User mentioned consultation but not explicitly requesting - providing general response...")
                    
                    # Generate general consultation mention response
                    consultation_response = self._generate_consultation_intent_response(explicit=False)
                    print(f"DEBUG: Generated consultation response: {consultation_response[:100]}...")
                    
                    processing_time = time.time() - start_time
                    
                    # Return the consultation response
                    return ChatResponse(
                        response=consultation_response,
                        session_id=session_id or "default",
                        confidence=0.8,  # Medium confidence for general consultation mention
                        processing_time=processing_time
                    )
            
            # If not a consultation request, proceed with normal RAG response
            # Get or create memory for this session
            memory = self.get_or_create_memory(session_id or "default")
            
            # Get conversation history
            chat_history = memory.chat_memory.messages
            
            # Create the prompt with conversation history
            if chat_history:
                # Build conversation context
                history_context = "Previous conversation:\n"
                for message in chat_history[-6:]:  # Last 6 messages
                    if hasattr(message, 'content'):
                        role = "Human" if message.__class__.__name__ == "HumanMessage" else "Assistant"
                        history_context += f"{role}: {message.content}\n"
                
                # Combine with current context
                full_context = f"{history_context}\nCurrent context: {context}\n\nUser Question: {query}"
            else:
                full_context = f"Context: {context}\n\nUser Question: {query}"
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=full_context)
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Format the response for better readability
            formatted_response = self._format_response(response.content)
            
            # Save to memory
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(formatted_response)
            
            processing_time = time.time() - start_time
            
            return ChatResponse(
                response=formatted_response,
                session_id=session_id or "default",
                confidence=0.8,  # Could be calculated based on search scores
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ChatResponse(
                response=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                session_id=session_id or "default",
                confidence=0.0,
                processing_time=processing_time
            )
    
    def chat(self, query: str, session_id: str = None, user_context: Dict[str, Any] = None) -> ChatResponse:
        """Main chat method that combines retrieval and generation"""
        # Retrieve relevant documents
        search_results = self.retrieve_relevant_documents(query)
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate response
        response = self.generate_response(query, context, session_id, user_context)
        
        # Don't include sources in the response
        return response
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation for a session"""
        if session_id in self.conversation_memories:
            memory = self.conversation_memories[session_id]
            messages = memory.chat_memory.messages
            
            return {
                "session_id": session_id,
                "message_count": len(messages),
                "topics_discussed": self._extract_topics(messages),
                "last_activity": messages[-1].content if messages else None,
                "conversation_history": [
                    {
                        "role": "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant",
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    for msg in messages[-10:]  # Last 10 messages
                ]
            }
        else:
            return {
                "session_id": session_id,
                "message_count": 0,
                "topics_discussed": [],
                "last_activity": None,
                "conversation_history": []
            }
    
    def _extract_topics(self, messages: List[BaseMessage]) -> List[str]:
        """Extract topics from conversation messages"""
        topics = []
        for message in messages:
            content = message.content.lower()
            if "pricing" in content or "cost" in content:
                topics.append("pricing")
            elif "service" in content or "offer" in content:
                topics.append("services")
            elif "demo" in content or "consultation" in content:
                topics.append("demo")
            elif "implementation" in content or "deploy" in content:
                topics.append("implementation")
        return list(set(topics))  # Remove duplicates
    
    def clear_conversation_memory(self, session_id: str) -> bool:
        """Clear conversation memory for a session"""
        if session_id in self.conversation_memories:
            del self.conversation_memories[session_id]
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs"""
        return list(self.conversation_memories.keys())
    
    def _extract_consultation_details(self, user_message: str) -> Optional[Dict[str, str]]:
        """Extract consultation details from user message"""
        # Look for patterns that indicate consultation details
        consultation_patterns = {
            'name': r'(?:name|i am|my name is|call me)\s*:?\s*([a-zA-Z\s]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'(?:phone|contact|call|number)\s*:?\s*([0-9\s\-\(\)\+]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
            'company': r'(?:company|firm|business|organization|work at|work for)\s*:?\s*([a-zA-Z0-9\s&.,]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
            'date': r'(?:date|schedule|appointment|meeting)\s*:?\s*([a-zA-Z0-9\s,]+?)(?:\s*,\s*|\s*and\s*|\s*\.|\s*$)',
            'time': r'(?:time|at|around)\s*:?\s*([0-9]+\s*(?:am|pm|AM|PM|[0-9:]+))',
            'message': r'(?:message|details|about|regarding|project)\s*:?\s*(.+)'
        }
        
        details = {}
        user_lower = user_message.lower()
        
        # Check if this looks like consultation details
        consultation_indicators = ['schedule', 'consultation', 'appointment', 'meeting', 'call', 'demo']
        if not any(indicator in user_lower for indicator in consultation_indicators):
            return None
        
        # Extract details using regex patterns
        for field, pattern in consultation_patterns.items():
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            if matches:
                details[field] = matches[0].strip()
        
        # Must have at least name and email to be considered a consultation request
        if 'name' in details and 'email' in details:
            return details
        
        return None
    
    def _schedule_consultation(self, details: Dict[str, str], session_id: str) -> Dict[str, Any]:
        """Schedule a consultation using the API"""
        try:
            # Prepare the consultation data
            consultation_data = {
                'name': details.get('name', ''),
                'email': details.get('email', ''),
                'phone': details.get('phone', ''),
                'company': details.get('company', ''),
                'preferred_date': details.get('date', ''),
                'preferred_time': details.get('time', ''),
                'message': details.get('message', '')
            }
            
            print(f"DEBUG: Calling API with data: {consultation_data}")
            
            # Call the consultation scheduling API
            response = requests.post(
                'http://localhost:8000/consultation/schedule',
                json=consultation_data,
                timeout=10
            )
            
            print(f"DEBUG: API response status: {response.status_code}")
            print(f"DEBUG: API response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'consultation_id': result.get('consultation_id', ''),
                    'message': result.get('message', ''),
                    'logged': result.get('logged', False)
                }
            else:
                return {
                    'success': False,
                    'error': f'API error: {response.status_code}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Error scheduling consultation: {str(e)}'
            }
    
    def _generate_consultation_response(self, details: Dict[str, str], schedule_result: Dict[str, Any]) -> str:
        """Generate response for consultation scheduling"""
        if schedule_result['success']:
            consultation_id = schedule_result.get('consultation_id', '')
            return f"""Perfect! I've successfully scheduled your consultation. Here are the details:

Consultation ID: {consultation_id}
Name: {details.get('name', '')}
Email: {details.get('email', '')}
Company: {details.get('company', 'N/A')}
Phone: {details.get('phone', 'N/A')}
Preferred Date: {details.get('date', 'To be confirmed')}
Preferred Time: {details.get('time', 'To be confirmed')}

Our team will contact you within 24 hours to confirm your appointment and discuss your specific needs.

You can also reach us directly at +1 (555) 012-3456 or ask@softtechniques.com for immediate assistance."""
        else:
            return f"""I've noted your consultation details, but there was a technical issue scheduling your appointment automatically.

Here's what you provided:
- Name: {details.get('name', '')}
- Email: {details.get('email', '')}
- Company: {details.get('company', 'N/A')}
- Phone: {details.get('phone', 'N/A')}

Please contact us directly at +1 (555) 012-3456 or ask@softtechniques.com to schedule your consultation, or try our scheduling form at /schedule-consultation."""
    
    def _format_response(self, response: str) -> str:
        """Format response for better readability and ChatGPT-like structure"""
        # Remove any markdown formatting
        response = response.replace("**", "").replace("*", "")
        
        # Convert numbered lists to bullet points
        response = re.sub(r'(\d+)\.\s*', '- ', response)
        
        # Handle cases where bullet points might be missing spaces or formatting
        response = re.sub(r'-\s*([^\s])', r'- \1', response)
        
        # Convert any remaining list patterns to bullet points
        response = re.sub(r'^(\s*)([•·▪▫])\s*', r'\1- ', response, flags=re.MULTILINE)
        
        # Ensure each bullet point starts on a new line with proper spacing
        # First, ensure bullets are properly formatted with space after dash
        response = re.sub(r'-([^\s])', r'- \1', response)
        
        # Ensure each bullet point starts on a new line
        response = re.sub(r'(?<!\n)\n- ', '\n\n- ', response)
        response = re.sub(r'^- ', '- ', response)  # Fix first bullet if needed
        
        # Ensure proper spacing between paragraphs and sections
        response = re.sub(r'(?<!\n)\n(?=[A-Z])', '\n\n', response)  # Add space before new paragraphs
        
        # Clean up excessive spacing (more than 2 newlines)
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure proper spacing around bullet lists
        response = re.sub(r'(?<!\n)\n- ', '\n\n- ', response)
        
        # Add spacing before and after bullet lists
        response = re.sub(r'(?<!\n)\n- ', r'\n\n- ', response)
        response = re.sub(r'(- .*?)(?=\n\n[A-Z])', r'\1\n', response)
        
        # Final cleanup: ensure consistent spacing
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Don't automatically add call-to-actions - let the LLM handle this naturally
        
        return response.strip()
    
    def _extract_consultation_details(self, query: str) -> Optional[Dict[str, str]]:
        """Extract consultation details from user query - only show form for explicit consultation requests"""
        # More specific consultation keywords that indicate user wants to schedule
        explicit_consultation_phrases = [
            "schedule a consultation", "book a consultation", "schedule consultation",
            "book consultation", "want to schedule", "want to book", "need consultation",
            "get consultation", "have consultation", "set up consultation",
            "arrange consultation", "plan consultation", "organize consultation"
        ]
        
        # General consultation keywords (less specific)
        general_keywords = [
            "consultation", "meeting", "appointment", "call", "demo", "discuss", "talk", "consult"
        ]
        
        query_lower = query.lower()
        
        # Check for explicit consultation scheduling requests
        if any(phrase in query_lower for phrase in explicit_consultation_phrases):
            return {"intent": "schedule_consultation", "explicit": True}
        
        # Check for general consultation mentions (but don't show form automatically)
        if any(keyword in query_lower for keyword in general_keywords):
            return {"intent": "consultation_mention", "explicit": False}
        
        return None
    
    def _generate_consultation_intent_response(self, explicit: bool = True) -> str:
        """Generate response when user wants to schedule a consultation"""
        if explicit:
            return """Perfect! I'd love to help you schedule a consultation to discuss your AI needs.

To book your consultation, please click the "Schedule Consultation" button below. The form will collect all the necessary information including:

- Your contact details
- Preferred date and time
- Your specific AI requirements
- Any questions you'd like to discuss

Our team will confirm your appointment within 24 hours and reach out to discuss how Soft Techniques can help transform your business with custom AI solutions.

Ready to get started? Click the "Schedule Consultation" button now! 🚀"""
        else:
            return """I'd be happy to discuss your AI needs! 

If you'd like to schedule a consultation to explore how Soft Techniques can help your business, you can use the scheduling form to book a time that works for you.

Our AI experts are ready to discuss your specific requirements and show you how our custom AI solutions can transform your business operations.

Feel free to ask me any questions about our services!"""
    
    def _schedule_consultation(self, details: Dict[str, str], session_id: str) -> Dict[str, Any]:
        """Handle consultation scheduling - now directs to form"""
        # Since we're directing to form, we don't actually schedule here
        # This method is kept for compatibility but now returns a form direction
        return {
            'success': True,
            'form_direction': True,
            'message': 'Please use the scheduling form to book your consultation'
        }

# Initialize RAG system
rag_system = RAGSystem()
