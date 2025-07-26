import logging
import os
from typing import Dict, Any, Optional
import time
import json

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid

from vapi_service import VapiService

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aven Support AI (MVP)")
vapi_service = VapiService()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
active_sessions: Dict[str, Dict] = {}
active_vapi_calls: Dict[str, Dict] = {}


class VapiWebhook(BaseModel):
    message: Dict[str, Any]


class ChatRequest(BaseModel):
    message: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    assistantId: Optional[str] = None
    session_id: Optional[str] = None
    response_time: Optional[float] = None


@app.get("/")
def read_root():
    return {"status": "Aven Support AI is running"}


@app.get("/health")
def health_check():
    """Detailed health check mimicking reference backend structure"""
    return {
        "status": "healthy",
        "agent_available": True,  # Simple check – can be extended
        "vapi_available": vapi_service.vapi is not None,
        "active_sessions": len(active_sessions),
        "active_vapi_calls": len(active_vapi_calls),
        "timestamp": time.time(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_handler(chat_request: ChatRequest):
    """
    Handles chat requests from the frontend.
    If a message is provided, processes it and returns a text response.
    Always returns the Vapi Assistant ID if available.
    """
    start_time = time.time()
    session_id = chat_request.session_id or str(uuid.uuid4())
    response_text = ""
    assistant_id = None

    try:
        # Process text message if provided
        if chat_request.message:
            response_text = await vapi_service.process_chat_message(
                chat_request.message, session_id
            )
            logger.info(f"Chat response generated for session {session_id}")

        # Try to get or create Vapi assistant
        try:
            assistant_id = await vapi_service.get_or_create_assistant()
            if not assistant_id:
                logger.error("get_or_create_assistant returned None. Vapi is likely unavailable.")
        except Exception as e:
            logger.error(f"Could not get/create Vapi assistant: {e}", exc_info=True)
            assistant_id = None # Ensure it's None on error

        # If we have no text response but Vapi failed, provide a fallback
        if not response_text and not assistant_id:
            response_text = "I'm having trouble connecting to the voice service. Please try again later or contact support@aven.com."

        response_time = time.time() - start_time
        return ChatResponse(
            response=response_text,
            assistantId=assistant_id,
            session_id=session_id,
            response_time=response_time
        )
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/vapi/webhook")
async def handle_vapi_webhook(request: Request):
    """
    Handles all Vapi webhooks with robust parsing, including tool calls.
    """
    try:
        payload = await request.json()
        logger.debug(f"Received Vapi webhook payload: {payload}")
        message = payload.get("message", {})
        
        if message.get("type") == "tool-calls":
            tool_calls = message.get("toolCalls", [])
            results = []

            for tool_call in tool_calls:
                function_name = tool_call.get("function", {}).get("name")
                tool_call_id = tool_call.get("id")

                # Robustly parse arguments
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                parameters = {}
                try:
                    if isinstance(arguments_str, str):
                        parameters = json.loads(arguments_str)
                    elif isinstance(arguments_str, dict):
                        parameters = arguments_str
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {arguments_str}", exc_info=True)

                logger.info(
                    f"Handling tool call: '{function_name}' with params: {parameters}"
                )

                result = await vapi_service.handle_tool_call(function_name, parameters)
                results.append({"toolCallId": tool_call_id, "result": result})

            return {"results": results}

        elif message.get("type") == "function-call":
            # Handle legacy function-call format if needed
            function_call_data = message.get("functionCall") or message.get("function_call", {})
            function_name = function_call_data.get("name")
            parameters = function_call_data.get("parameters", {})
            logger.info(f"Handling legacy function-call: '{function_name}' with params: {parameters}")
            result = await vapi_service.handle_tool_call(function_name, parameters)
            return {"result": result}
        
        # Add handling for other webhook types here if needed (e.g., status-update)
        logger.info(f"Received webhook of type '{message.get('type')}', no action taken.")
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Error in Vapi webhook handler: {e}", exc_info=True)
        # Return a 200 OK to prevent Vapi from retrying on a failing webhook
        return {"status": "error", "message": str(e)}, 200


# DEPRECATED - consolidated into /vapi/webhook
# @app.post("/vapi-webhook")
# async def handle_vapi_webhook_deprecated(request: Request):
#     # ... implementation removed ...

# DEPRECATED - consolidated into /vapi/webhook
# @app.post("/vapi/function-call")
# async def vapi_function_call_deprecated(request: Request):
#     # ... implementation removed ...


class VapiCallRequest(BaseModel):
    phone_number: Optional[str] = None  # Add phone number for phone calls
    type: Optional[str] = "web"  # Only web for now

# Vapi Integration Endpoints
@app.post("/vapi/assistant")
async def create_vapi_assistant():
    """Create a Vapi assistant and return its ID"""
    try:
        assistant_id = await vapi_service.get_or_create_assistant()
        if not assistant_id:
            logger.error("Failed to get assistant ID from vapi_service.")
            raise HTTPException(
                status_code=503,
                detail="Service Unavailable: Could not retrieve Vapi assistant ID. Check VAPI_API_KEY and backend logs."
            )
        return {"assistant_id": assistant_id}
    except Exception as e:
        logger.error(f"Error creating Vapi assistant: {e}", exc_info=True)
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
        logger.error(f"Error creating Vapi call: {e}", exc_info=True)
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
        logger.error(f"Error getting call status: {e}", exc_info=True)
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
        logger.error(f"Error ending call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to end call: {str(e)}")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 