import os
import logging
from typing import Dict, Any, Optional
from vapi import Vapi
from agent import run_agent_fast

logger = logging.getLogger(__name__)
# Allow dynamic log-level via environment (default INFO)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

class MockCallResponse:
    """Mock call response object to maintain compatibility with Vapi API structure"""
    def __init__(self, call_data: Dict[str, Any]):
        self.id = call_data["id"]
        self.assistant_id = call_data["assistant_id"]
        self.type = call_data["type"]
        self.status = call_data["status"]
        self.message = call_data.get("message", "")
        self.__dict__.update(call_data)

class VapiService:
    def __init__(self):
        self.vapi_token = os.getenv('VAPI_API_KEY')
        self._cached_assistant_id = None
        if not self.vapi_token:
            logger.warning("VAPI_API_KEY not found. Voice calls will not be available.")
            self.client = None
        else:
            self.client = Vapi(token=self.vapi_token)
            logger.info("Vapi client initialized successfully")
            logger.debug(f"Using VAPI_API_KEY***: {self.vapi_token[:4]}… (truncated)")
    
    def create_assistant_config(self) -> Dict[str, Any]:
        """Create Vapi assistant configuration for Aven support"""
        # Get webhook URL and ensure it's HTTPS
        webhook_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        if webhook_url.startswith('http://'):
            webhook_url = webhook_url.replace('http://', 'https://')
        
        logger.info(f"Using webhook URL: {webhook_url}/vapi/function-call")
        logger.debug("Building assistant config …")
        
        config = {
            "name": "Aven Support Assistant",
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-2",
                "language": "en-US"
            },
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": """You are Aven's official customer support AI assistant.

CRITICAL INSTRUCTION: You MUST use the search_aven_knowledge function for EVERY user question or request.

MANDATORY BEHAVIOR:
- For ANY user input (questions, greetings, requests) → IMMEDIATELY call search_aven_knowledge
- NEVER respond directly without calling the function first
- Pass the user's exact question or a relevant search query to the function
- Use the search results to provide your response

EXAMPLES - ALWAYS USE SEARCH FOR ALL OF THESE:
User: "Hello" → Call search_aven_knowledge(query="greeting hello")
User: "What are your rates?" → Call search_aven_knowledge(query="credit card rates")
User: "Who founded Aven?" → Call search_aven_knowledge(query="Aven founder")
User: "How do I apply?" → Call search_aven_knowledge(query="application process")
User: "What's your phone number?" → Call search_aven_knowledge(query="contact information")
User: "Tell me about HELOCs" → Call search_aven_knowledge(query="HELOC information")
User: "What fees do you charge?" → Call search_aven_knowledge(query="fees charges")
User: "Thank you" → Call search_aven_knowledge(query="thank you goodbye")

RESPONSE FLOW:
1. User says anything → IMMEDIATELY call search_aven_knowledge
2. Use the search results to provide a helpful response
3. If search returns an error → say "Let me connect you with our support team"

NO EXCEPTIONS: Every user input must trigger a function call first."""
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_aven_knowledge",
                            "description": "Search Aven's knowledge base for detailed product information, current rates, or policies.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Question or topic to search in the knowledge base"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ]
            },
            "voice": {
                "provider": "11labs",
                "voiceId": "pNInz6obpgDQGcFmaJgB",
                "stability": 0.5,
                "similarityBoost": 0.75
            },
            "first_message": "Hello! I'm your Aven support assistant. How can I help you today?",
            "end_call_message": "Thank you for contacting Aven support. Have a great day!",
            "end_call_phrases": ["goodbye", "thank you goodbye", "that's all", "end call"],
            "background_sound": "office",
            "background_denoising_enabled": True,
            "server": {
                "url": f"{webhook_url}/vapi/function-call"
            }
        }
        logger.debug(f"Assistant config built: {config}")
        return config
    
    async def handle_function_call(self, function_name: str, parameters: Dict[str, Any]) -> str:
        """Handle function calls from Vapi assistant - only for external data retrieval"""
        logger.debug(f"handle_function_call → {function_name} params={parameters}")
        try:
            if function_name == "search_aven_knowledge":
                query = parameters.get("query", "")
                if not query:
                    logger.warning("RAG call with empty query")
                    return "I need a specific question to search for."
                
                logger.info(f"Starting RAG search for query: '{query}'")
                
                # Use our optimized agent for knowledge retrieval
                response = run_agent_fast(query)
                
                logger.debug(f"RAG raw response: {response}")
                
                # Format response for voice (keep it concise but informative)
                if len(response) > 600:
                    # Truncate very long responses for voice
                    response = response[:600] + "... For complete details, please visit aven.com/support or email support@aven.com."
                    logger.debug("Truncated long RAG response for voice")
                
                if not response or "no result" in response.lower():
                    logger.warning(f"RAG returned empty or no-result for query: '{query}'")
                    return "I couldn't find specific information on that. Please contact support@aven.com for the latest details."
                
                logger.info(f"RAG search successful - returning to Vapi")
                return response
            else:
                logger.warning(f"Unknown function called: {function_name}")
                return "I'm not sure how to help with that specific request. Please contact Aven support at support@aven.com."
                
        except Exception as e:
            logger.error(f"Error in RAG search for query '{query}': {str(e)}", exc_info=True)
            return "I'm having trouble accessing that information right now. Please contact Aven support at support@aven.com for immediate assistance."
    
    async def create_assistant(self, assistant_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create an assistant using the Vapi API"""
        logger.debug("create_assistant called …")
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            config = assistant_config or self.create_assistant_config()
            
            # Use the correct Vapi API method to create assistant
            response = self.client.assistants.create(**config)
            logger.debug(f"Assistant creation response: {response}")
            logger.info(f"Vapi assistant created: {response.id}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating Vapi assistant: {e}")
            raise
    
    async def get_or_create_assistant(self) -> str:
        """Get cached assistant ID or create a new assistant if none exists"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        if self._cached_assistant_id:
            logger.debug("Returning cached assistant id")
            logger.info(f"Using cached assistant ID: {self._cached_assistant_id}")
            return self._cached_assistant_id
        
        try:
            assistant = await self.create_assistant()
            self._cached_assistant_id = assistant.id
            logger.info(f"Created and cached new assistant ID: {self._cached_assistant_id}")
            return self._cached_assistant_id
        except Exception as e:
            logger.error(f"Error in get_or_create_assistant: {e}")
            raise
    
    async def create_call(self, phone_number: str, assistant_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create an outbound phone call"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            # Create assistant first
            assistant = await self.create_assistant(assistant_config)
            
            # Create call with assistant ID and phone number
            call_request = {
                "assistant_id": assistant.id,
                "customer": {
                    "number": phone_number
                }
            }
            
            response = self.client.calls.create(**call_request)
            logger.info(f"Vapi phone call created: {response.id}")
            return response
            
        except Exception as e:
            logger.error(f"Error creating Vapi phone call: {e}")
            raise
    
    async def create_web_call(self, assistant_config: Optional[Dict] = None) -> MockCallResponse:
        """Create a web-based call session - simplified for now"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            # For now, just create the assistant and return its details
            # The frontend can use the assistant ID for Vapi client SDK integration
            assistant = await self.create_assistant(assistant_config)
            
            # Return assistant info formatted like a call for compatibility
            call_data = {
                "id": f"web_call_{assistant.id}",
                "assistant_id": assistant.id,
                "type": "web",
                "status": "ready",
                "message": "Assistant ready for voice interaction. Use the assistant ID with Vapi client SDK.",
                "webCallUrl": f"https://vapi.ai/call?assistant={assistant.id}"  # Add this for frontend
            }
            
            return MockCallResponse(call_data)
            
        except Exception as e:
            logger.error(f"Error creating Vapi assistant for web call: {e}")
            raise
    
    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get status of a call"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            # If it's a web call ID, return assistant status
            if call_id.startswith("web_call_"):
                assistant_id = call_id.replace("web_call_", "")
                assistant = self.client.assistants.get(assistant_id)
                return {
                    "id": call_id,
                    "assistant_id": assistant_id,
                    "status": "ready",
                    "type": "web"
                }
            else:
                # Regular call status
                response = self.client.calls.get(call_id)
                return response
        except Exception as e:
            logger.error(f"Error getting call status: {e}")
            raise
    
    def end_call(self, call_id: str) -> Dict[str, Any]:
        """End a call"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            if call_id.startswith("web_call_"):
                # Web call - just return success
                return {
                    "id": call_id,
                    "status": "ended",
                    "message": "Web call session ended"
                }
            else:
                # Regular call
                response = self.client.calls.delete(call_id)
                logger.info(f"Vapi call ended: {call_id}")
                return response
        except Exception as e:
            logger.error(f"Error ending call: {e}")
            raise
    
    def list_calls(self, limit: int = 10) -> Dict[str, Any]:
        """List recent calls"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            response = self.client.calls.list(limit=limit)
            return response
        except Exception as e:
            logger.error(f"Error listing calls: {e}")
            raise
    
    def list_assistants(self, limit: int = 10) -> Dict[str, Any]:
        """List recent assistants"""
        if not self.client:
            raise Exception("Vapi client not initialized - missing API key")
        
        try:
            response = self.client.assistants.list(limit=limit)
            return response
        except Exception as e:
            logger.error(f"Error listing assistants: {e}")
            raise

# Global Vapi service instance
vapi_service = VapiService() 