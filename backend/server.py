import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from agent import run_agent_fast, llm
from vapi_service import vapi_service
import uuid
from contextlib import asynccontextmanager
from vapi_service import vapi_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Aven Support API", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https:\/\/.*\.ngrok(?:-free)?\.(?:io|app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    response_time: float
    source_type: str = "agent"

class VoiceMessage(BaseModel):
    type: str  # "start_recording", "stop_recording", "audio_data", "text_input"
    data: Optional[str] = None
    session_id: Optional[str] = None

class VapiCallRequest(BaseModel):
    phone_number: Optional[str] = None
    type: str = "web"  # "web" or "phone"

# Vapi webhook models based on documentation
class VapiFunctionCallMessage(BaseModel):
    type: str = "function-call"
    functionCall: Dict[str, str]
    call: Optional[Dict] = None

# Session management
active_sessions: Dict[str, Dict] = {}
active_vapi_calls: Dict[str, Dict] = {}

def create_session(session_id: str = None) -> str:
    """Create a new chat session"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = {
        "created_at": time.time(),
        "messages": [],
        "last_activity": time.time()
    }
    
    logger.info(f"Created new session: {session_id}")
    return session_id

def update_session_activity(session_id: str):
    """Update last activity timestamp for session"""
    if session_id in active_sessions:
        active_sessions[session_id]["last_activity"] = time.time()

def add_message_to_session(session_id: str, message: str, sender: str):
    """Add message to session history"""
    if session_id in active_sessions:
        active_sessions[session_id]["messages"].append({
            "content": message,
            "sender": sender,
            "timestamp": time.time()
        })

# HTTP Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Aven Support API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agent_available": llm is not None,
        "vapi_available": vapi_service.client is not None,
        "active_sessions": len(active_sessions),
        "active_vapi_calls": len(active_vapi_calls),
        "timestamp": time.time()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """HTTP endpoint for text-based chat"""
    try:
        start_time = time.time()
        
        # Create or get session
        session_id = chat_request.session_id or create_session()
        update_session_activity(session_id)
        
        # Add user message to session
        add_message_to_session(session_id, chat_request.message, "user")
        
        # Get response from agent
        logger.info(f"Processing chat request in session {session_id}: {chat_request.message}")
        response = run_agent_fast(chat_request.message)
        
        # Add agent response to session
        add_message_to_session(session_id, response, "agent")
        
        response_time = time.time() - start_time
        
        logger.info(f"Chat response generated in {response_time:.2f}s for session {session_id}")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            response_time=response_time,
            source_type="agent"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information and message history"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "last_activity": session["last_activity"],
        "message_count": len(session["messages"]),
        "messages": session["messages"][-10:]  # Return last 10 messages
    }

# Vapi Integration Endpoints
@app.post("/vapi/assistant")
async def create_vapi_assistant():
    """Create a Vapi assistant and return its ID"""
    try:
        assistant_id = await vapi_service.get_or_create_assistant()
        return {"assistant_id": assistant_id}
    except Exception as e:
        logger.error(f"Error creating Vapi assistant: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")

@app.post("/vapi/call")
async def create_vapi_call(call_request: VapiCallRequest):
    """Create a Vapi voice call"""
    try:
        if call_request.type == "phone" and call_request.phone_number:
            # Create phone call
            call_response = await vapi_service.create_call(call_request.phone_number)
        else:
            # Create web call
            call_response = await vapi_service.create_web_call()
        
        # Store call info
        call_id = call_response.id
        active_vapi_calls[call_id] = {
            "created_at": time.time(),
            "type": call_request.type,
            "phone_number": call_request.phone_number,
            "status": "active"
        }
        
        logger.info(f"Vapi call created: {call_id}")
        return {
            "success": True,
            "call_id": call_id,
            "call_response": call_response.__dict__ if hasattr(call_response, '__dict__') else call_response
        }
        
    except Exception as e:
        logger.error(f"Error creating Vapi call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create call: {str(e)}")

@app.get("/vapi/call/{call_id}")
async def get_vapi_call_status(call_id: str):
    """Get Vapi call status"""
    try:
        status = vapi_service.get_call_status(call_id)
        return {
            "success": True,
            "call_id": call_id,
            "status": status.__dict__ if hasattr(status, '__dict__') else status
        }
    except Exception as e:
        logger.error(f"Error getting call status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get call status: {str(e)}")

@app.post("/vapi/call/{call_id}/end")
async def end_vapi_call(call_id: str):
    """End a Vapi call"""
    try:
        result = vapi_service.end_call(call_id)
        
        # Update local call status
        if call_id in active_vapi_calls:
            active_vapi_calls[call_id]["status"] = "ended"
            active_vapi_calls[call_id]["ended_at"] = time.time()
        
        logger.info(f"Vapi call ended: {call_id}")
        return {
            "success": True,
            "call_id": call_id,
            "result": result.__dict__ if hasattr(result, '__dict__') else result
        }
    except Exception as e:
        logger.error(f"Error ending call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end call: {str(e)}")

@app.post("/vapi/function-call")
async def handle_vapi_function_call(request: Request):
    """Handle function calls and other webhooks from the Vapi assistant."""
    try:
        body = await request.json()
        logger.info(f"🔔 Full Vapi webhook body: {body}")

        message = body.get("message", {})
        webhook_type = message.get("type")

        # This handles direct test invocations of the function-call endpoint
        if webhook_type == "function-call":
            function_call_data = message.get("functionCall", {})
            if not function_call_data:
                logger.warning("Function-call webhook with no functionCall payload.")
                return {"received": True}
                 
            function_name = function_call_data.get("name")
            parameters = function_call_data.get("parameters", {})
            logger.info(f"⚙️  Direct function call '{function_name}' with params {parameters}")
            result = await vapi_service.handle_function_call(function_name, parameters)
            return {"result": result}
        
        # Handle the 'tool-calls' event type specifically
        elif webhook_type == "tool-calls":
            logger.info("⚙️ Received 'tool-calls' webhook, processing now.")
            
            tool_calls = message.get("toolCalls", [])
            if not tool_calls:
                logger.warning("No tool calls found in tool-calls webhook")
                return {"received": True}
            
            # Process each tool call and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function_data = tool_call.get("function", {})
                function_name = function_data.get("name")
                
                # Parse arguments (handle both string and dict)
                arguments_str = function_data.get("arguments")
                if isinstance(arguments_str, str):
                    try:
                        parameters = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {e}")
                        parameters = {}
                elif isinstance(arguments_str, dict):
                    parameters = arguments_str
                else:
                    logger.warning(f"Unexpected arguments type: {type(arguments_str)}")
                    parameters = {}
                
                logger.info(f"Executing tool call '{function_name}' with ID '{tool_call_id}' and params {parameters}")
                
                # Execute the function
                if function_name == "search_aven_knowledge":
                    result = await vapi_service.handle_function_call(function_name, parameters)
                    logger.info(f"Tool call '{function_name}' executed. Result (truncated): {result[:100]}...")
                    
                    tool_results.append({
                        "toolCallId": tool_call_id,
                        "result": result
                    })
                else:
                    logger.warning(f"Unknown function name: {function_name}")
                    tool_results.append({
                        "toolCallId": tool_call_id,
                        "result": f"Error: Unknown function '{function_name}'"
                    })
            
            # Return results to Vapi
            return {"results": tool_results}

        # This handles tool calls that come as part of a conversation stream
        elif webhook_type == "conversation-update":
            logger.info("📡 Received 'conversation-update' webhook.")
            conversation = message.get("conversation", [])
            
            # Iterate backwards to find the last assistant message with a tool call
            for i in range(len(conversation) - 1, -1, -1):
                msg = conversation[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tool_calls = msg.get("tool_calls", [])
                    
                    # Check if this tool call has already been answered
                    if i + 1 < len(conversation) and conversation[i+1].get("role") == "tool":
                        logger.debug(f"Tool call {tool_calls[0].get('id')} already has a result. Skipping.")
                        continue
                    
                    # It's an unanswered tool call, so let's execute it.
                    tool_call = tool_calls[0]
                    tool_call_id = tool_call.get("id")
                    function_details = tool_call.get("function", {})
                    function_name = function_details.get("name")
                    
                    try:
                        parameters = json.loads(function_details.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse arguments in conversation-update: {function_details.get('arguments')}")
                        parameters = {}

                    logger.info(f"Executing unanswered tool call from conversation history: '{function_name}' with ID '{tool_call_id}'")
                    result = await vapi_service.handle_function_call(function_name, parameters)
                    logger.info(f"Tool call '{function_name}' executed. Result (truncated): {str(result)[:200]}...")
                    
                    return {"tool_call_result": {"id": tool_call_id, "result": result}}
            
            logger.info("✅ 'conversation-update' processed. No new tool calls to answer.")
            return {"received": True}
        
        else:
            logger.info(f"📡 Received Vapi webhook type: {webhook_type or 'unknown'}. No action taken.")
            return {"received": True}

    except Exception as e:
        logger.error(f"💥 Error in Vapi webhook handler: {e}", exc_info=True)
        # Return a 200 to prevent Vapi from retrying a failing webhook.
        return {"status": "error", "message": str(e)}

@app.get("/vapi/calls")
async def list_active_vapi_calls():
    """List all active Vapi calls"""
    try:
        # Get recent calls from Vapi API
        vapi_calls = vapi_service.list_calls(limit=20)
        
        return {
            "active_calls": active_vapi_calls,
            "recent_vapi_calls": vapi_calls.__dict__ if hasattr(vapi_calls, '__dict__') else vapi_calls
        }
    except Exception as e:
        logger.error(f"Error listing calls: {e}")
        return {
            "active_calls": active_vapi_calls,
            "recent_vapi_calls": [],
            "error": str(e)
        }

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_sessions: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_sessions[websocket] = session_id
        
        # Create session if it doesn't exist
        if session_id not in active_sessions:
            create_session(session_id)
        
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_sessions:
            session_id = self.connection_sessions[websocket]
            del self.connection_sessions[websocket]
            logger.info(f"WebSocket disconnected for session: {session_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def send_to_session(self, message: dict, session_id: str):
        for connection in self.active_connections:
            if self.connection_sessions.get(connection) == session_id:
                await self.send_personal_message(message, connection)

manager = ConnectionManager()

# WebSocket endpoint for voice chat (now enhanced with Vapi option)
@app.websocket("/ws/voice/{session_id}")
async def voice_websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        # Send welcome message with Vapi option
        await manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Voice chat connected. You can use text input or start a Vapi voice call.",
            "vapi_available": vapi_service.client is not None
        }, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            content = message_data.get("data", "")
            
            logger.info(f"Received WebSocket message type: {message_type} in session: {session_id}")
            
            if message_type == "text_input":
                # Handle text input through WebSocket
                await handle_text_input(websocket, session_id, content)
                
            elif message_type == "start_vapi_call":
                # Start Vapi voice call
                try:
                    assistant_id = await vapi_service.get_or_create_assistant()
                    
                    await manager.send_personal_message({
                        "type": "vapi_call_created",
                        "call_id": f"web_sdk_{assistant_id}",
                        "assistant_id": assistant_id,
                        "message": "Vapi assistant ready for Web SDK integration"
                    }, websocket)
                    
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Failed to create Vapi call: {str(e)}"
                    }, websocket)
            
            elif message_type == "voice_start":
                # Voice recording started (legacy support)
                await manager.send_personal_message({
                    "type": "voice_status",
                    "status": "listening",
                    "message": "Listening... Speak now or consider using Vapi for better voice experience."
                }, websocket)
                
            elif message_type == "voice_end":
                # Voice recording ended (legacy support)
                await manager.send_personal_message({
                    "type": "voice_status", 
                    "status": "processing",
                    "message": "Processing your voice input..."
                }, websocket)
                
                # Simulate voice processing
                await asyncio.sleep(1)
                simulated_text = content or "For better voice experience, please use the Vapi voice call feature."
                
                await handle_text_input(websocket, session_id, simulated_text)
                
            elif message_type == "ping":
                # Handle ping for connection keepalive
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": time.time()
                }, websocket)
                
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error in session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "An error occurred. Please refresh and try again."
        }, websocket)
        manager.disconnect(websocket)

async def handle_text_input(websocket: WebSocket, session_id: str, user_input: str):
    """Handle text input from WebSocket (voice or text)"""
    try:
        start_time = time.time()
        
        # Update session activity
        update_session_activity(session_id)
        add_message_to_session(session_id, user_input, "user")
        
        # Send acknowledgment
        await manager.send_personal_message({
            "type": "message_received",
            "content": user_input,
            "sender": "user",
            "timestamp": time.time()
        }, websocket)
        
        # Send processing status
        await manager.send_personal_message({
            "type": "agent_typing",
            "message": "Agent is thinking..."
        }, websocket)
        
        # Get response from agent
        logger.info(f"Processing WebSocket message in session {session_id}: {user_input}")
        response = run_agent_fast(user_input)
        
        # Add agent response to session
        add_message_to_session(session_id, response, "agent")
        
        response_time = time.time() - start_time
        
        # Send response
        await manager.send_personal_message({
            "type": "agent_response",
            "content": response,
            "sender": "agent", 
            "timestamp": time.time(),
            "response_time": response_time,
            "session_id": session_id
        }, websocket)
        
        logger.info(f"WebSocket response sent in {response_time:.2f}s for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error handling text input in session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": "Sorry, I encountered an error processing your request. Please try again."
        }, websocket)

# Cleanup inactive sessions periodically
@app.on_event("startup")
async def startup_event():
    async def cleanup_sessions():
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            current_time = time.time()
            inactive_sessions = []
            inactive_calls = []
            
            # Clean up sessions
            for session_id, session_data in active_sessions.items():
                if current_time - session_data["last_activity"] > 3600:  # 1 hour timeout
                    inactive_sessions.append(session_id)
            
            # Clean up old Vapi calls
            for call_id, call_data in active_vapi_calls.items():
                if current_time - call_data["created_at"] > 7200:  # 2 hour timeout
                    inactive_calls.append(call_id)
            
            for session_id in inactive_sessions:
                del active_sessions[session_id]
                logger.info(f"Cleaned up inactive session: {session_id}")
            
            for call_id in inactive_calls:
                del active_vapi_calls[call_id]
                logger.info(f"Cleaned up old Vapi call: {call_id}")
    
    # Start cleanup task
    asyncio.create_task(cleanup_sessions())

if __name__ == "__main__":
    logger.info("Starting Aven Support API server with Vapi integration...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 