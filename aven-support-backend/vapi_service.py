import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from vapi import Vapi

from mcp_tools import (
    RAGTool,
    SerperTool,
    CalendarTool,
    list_assistants,
    get_or_create_assistant,
    get_system_prompt,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.vapi_token = os.getenv("VAPI_API_KEY")
        self._cached_assistant_id: Optional[str] = None
        if not self.vapi_token:
            logger.warning("VAPI_API_KEY not found. Voice calls will not be available.")
            self.vapi = None
        else:
            self.vapi = Vapi(token=self.vapi_token)
            logger.info("Vapi client initialized successfully")
        
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag_tool = RAGTool()
        self.serper_tool = SerperTool()
        self.calendar_tool = CalendarTool()
        # In-memory session history for the text-based chat agent
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}

    async def handle_tool_call(
        self, function_name: str, parameters: Dict[str, Any]
    ) -> Any:
        logger.debug(f"Handling tool call: '{function_name}' with parameters: {parameters}")
        try:
            if function_name == "search_aven_knowledge":
                return await self.rag_tool.use(query=parameters.get("query"))
            elif function_name == "search_web":
                return await self.serper_tool.use(query=parameters.get("query"))
            elif function_name == "schedule_meeting":
                return await self.calendar_tool.schedule(
                    email=parameters.get("email"),
                    preferred_date=parameters.get("preferred_date"),
                    preferred_time=parameters.get("preferred_time"),
                )
            elif function_name == "check_availability":
                return await self.calendar_tool.check_availability(
                    date=parameters.get("date"), time=parameters.get("time")
                )
            else:
                logger.warning(f"Unknown tool: {function_name}")
                return {"error": "Unknown tool"}
        except Exception as e:
            logger.error(f"Error handling tool call {function_name}: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_or_create_assistant(self) -> Optional[str]:
        """Get cached assistant ID or create a new one."""
        if not self.vapi:
            logger.error("Vapi client not initialized")
            return None
        
        if self._cached_assistant_id:
            logger.debug(f"Returning cached assistant ID: {self._cached_assistant_id}")
            return self._cached_assistant_id

        try:
            assistant = await get_or_create_assistant(self.vapi, self.get_tools_schema())
            if assistant:
                self._cached_assistant_id = assistant.id
                logger.info(f"Cached new assistant ID: {self._cached_assistant_id}")
                return self._cached_assistant_id
            return None
        except Exception as e:
            logger.error(f"Error in get_or_create_assistant: {e}", exc_info=True)
            raise

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            self.rag_tool.schema,
            self.serper_tool.schema,
            self.calendar_tool.schedule_schema,
            self.calendar_tool.availability_schema,
        ]
        
    async def process_chat_message(self, message: str, session_id: str) -> str:
        """
        Processes a text chat message using a tool-calling agent loop.
        """
        logger.debug(f"Processing chat for session {session_id}: '{message}'")

        # Initialize or retrieve conversation history
        if session_id not in self.session_history:
            self.session_history[session_id] = [
                {"role": "system", "content": get_system_prompt()}
            ]
        
        # Add user's message to history
        self.session_history[session_id].append({"role": "user", "content": message})

        try:
            for _ in range(5): # Limit to 5 tool-calling iterations to prevent loops
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.session_history[session_id],
                    tools=self.get_tools_schema(),
                    tool_choice="auto",
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if not tool_calls:
                    # No tool calls, this is the final answer
                    final_answer = response_message.content
                    self.session_history[session_id].append({"role": "assistant", "content": final_answer})
                    logger.debug(f"Final answer for session {session_id}: {final_answer}")
                    return final_answer

                # Execute tool calls
                self.session_history[session_id].append(response_message)
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Text agent calling tool: {function_name} with args: {function_args}")
                    
                    function_response = await self.handle_tool_call(
                        function_name=function_name,
                        parameters=function_args,
                    )
                    
                    self.session_history[session_id].append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(function_response),
                        }
                    )

            # If the loop finishes, return a fallback message
            return "I seem to be having trouble processing that request. Please try rephrasing or contact support."

        except Exception as e:
            logger.error(f"Error processing chat message for session {session_id}: {e}", exc_info=True)
            return "I'm experiencing technical difficulties. Please try again later."

    async def create_web_call(self) -> MockCallResponse:
        """Create a web call session and return assistant info."""
        logger.debug("Creating web call...")
        if not self.vapi:
            raise Exception("Vapi client not initialized")
        
        assistant_id = await self.get_or_create_assistant()
        if not assistant_id:
            raise Exception("Failed to get or create assistant for web call")
            
        call_data = {
            "id": f"web_call_{assistant_id}",
            "assistant_id": assistant_id,
            "type": "web",
            "status": "ready",
            "message": "Assistant ready for voice interaction.",
            "webCallUrl": f"https://vapi.ai/call?assistant={assistant_id}"
        }
        logger.debug(f"Web call data created: {call_data}")
        return MockCallResponse(call_data)

    async def create_call(self, phone_number: str) -> Dict[str, Any]:
        """Create an outbound phone call."""
        if not self.vapi:
            raise Exception("Vapi client not initialized")
        
        assistant_id = await self.get_or_create_assistant()
        if not assistant_id:
            raise Exception("Failed to get or create assistant for phone call")
            
        return self.vapi.calls.create(
            assistant_id=assistant_id,
            customer={"number": phone_number}
        )

    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get status of a call."""
        if not self.vapi:
            raise Exception("Vapi client not initialized")
        return self.vapi.calls.get(call_id)

    def end_call(self, call_id: str) -> Dict[str, Any]:
        """End a call."""
        if not self.vapi:
            raise Exception("Vapi client not initialized")
        return self.vapi.calls.delete(call_id)

    def list_calls(self, limit: int = 20) -> Dict[str, Any]:
        """List recent calls."""
        if not self.vapi:
            raise Exception("Vapi client not initialized")
        return self.vapi.calls.list(limit=limit) 