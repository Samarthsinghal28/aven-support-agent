import asyncio
import logging
import os
from typing import Dict, Any, List

import aiofiles
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from playwright.async_api import async_playwright
from readability import Document as ReadabilityDocument
from vapi import Vapi
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta, date
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGTool:
    def __init__(self):
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = (
            self.pinecone.Index(self.index_name)
            if self.index_name in self.pinecone.list_indexes().names()
            else None
        )
        self.schema = {
            "type": "function",
            "function": {
                "name": "search_aven_knowledge",
                "description": "Search Aven's knowledge base for product information, policies, rates, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's question or topic to search for.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    async def use(self, query: str) -> Dict[str, Any]:
        logger.debug(f"RAGTool received query: '{query}'")
        if not query or not isinstance(query, str):
            logger.warning(f"RAGTool.use received an invalid query: {query}")
            return {"error": "A valid query is required to search the knowledge base."}
        if not self.index:
            return {"error": "Pinecone index not found."}

        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Add a timeout to the Pinecone query
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.index.query,
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                ),
                timeout=10.0  # 10-second timeout
            )
            logger.debug(f"Pinecone query results: {results}")
            
            contexts = []
            for match in results["matches"]:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                
                # Format the text nicely if it has a structured format
                if "Section:" in text and "Question:" in text and "Answer:" in text:
                    # Keep the structure but clean up the formatting
                    contexts.append(text)
                else:
                    # For unstructured text, just add it as is
                    contexts.append(text)
            
            # Clean up and format the response
            formatted_contexts = []
            for context in contexts:
                # Remove excessive whitespace
                context = " ".join(context.split())
                # Add proper paragraph breaks
                context = context.replace("Section:", "\nSection:")
                context = context.replace("Question:", "\nQuestion:")
                context = context.replace("Answer:", "\nAnswer:")
                formatted_contexts.append(context)
            
            return {
                "contexts": formatted_contexts
            }
        except asyncio.TimeoutError:
            logger.error("Pinecone query timed out.")
            return {"error": "The knowledge base search took too long to respond. Please try again."}
        except Exception as e:
            logger.error(f"Error querying RAG tool: {e}", exc_info=True)
            return {"error": str(e)}


class SerperTool:
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.schema = {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for recent or time-sensitive information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    async def use(self, query: str) -> Dict[str, Any]:
        logger.debug(f"SerperTool received query: '{query}'")
        url = "https://google.serper.dev/search"
        payload = {"q": query}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            search_results = response.json()
            logger.debug(f"Serper API raw response: {search_results}")
            
            # Format the results for better readability
            formatted_results = {
                "organic": [],
                "answerBox": search_results.get("answerBox", {}),
            }
            
            # Process organic results
            for result in search_results.get("organic", [])[:5]:  # Limit to top 5 results
                formatted_results["organic"].append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying Serper API: {e}", exc_info=True)
            return {"error": str(e)}


class CalendarTool:
    def __init__(self):
        self.credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "credentials.json")
        self.token_path = "token.json"
        self.service = self._initialize_google_calendar()
        self.schedule_schema = {
            "type": "function",
            "function": {
                "name": "schedule_meeting",
                "description": "Schedule a meeting with an Aven specialist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "User's email."},
                        "preferred_date": {
                            "type": "string",
                            "description": "Preferred date (YYYY-MM-DD).",
                        },
                        "preferred_time": {
                            "type": "string",
                            "description": "Preferred time (HH:MM 24-hour).",
                        },
                    },
                    "required": [
                        "email",
                        "preferred_date",
                        "preferred_time",
                    ],
                },
            },
        }
        self.availability_schema = {
            "type": "function",
            "function": {
                "name": "check_availability",
                "description": "Check availability for a meeting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date (YYYY-MM-DD)."},
                        "time": {"type": "string", "description": "Time (HH:MM 24-hour)."},
                    },
                    "required": ["date", "time"],
                },
            },
        }

    def _initialize_google_calendar(self):
        creds = None
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path)
            except Exception as e:
                logger.error(f"Failed to load credentials from {self.token_path}: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing Google Calendar token...")
                    creds.refresh(Request())
                except RefreshError as e:
                    logger.error(f"Token refresh failed: {e}. The refresh token is likely expired or revoked. Please re-authenticate.")
                    logger.error("ACTION REQUIRED: Run 'python setup_google_calendar.py' to generate a new token.json.")
                    return None
                except Exception as e:
                    logger.error(f"An unexpected error occurred during token refresh: {e}", exc_info=True)
                    return None
                else:
                    # Save the refreshed credentials
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                    logger.info("Token refreshed and saved successfully.")
            else:
                logger.warning(f"'{self.token_path}' not found or invalid. Please run 'python setup_google_calendar.py' to authorize.")
                return None
        
        try:
            return build("calendar", "v3", credentials=creds)
        except Exception as e:
            logger.error(f"Failed to build Google Calendar service: {e}", exc_info=True)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(HttpError),
        reraise=True  # Reraise the exception if all retries fail
    )
    async def schedule(
        self, email: str, preferred_date: str, preferred_time: str
    ):
        logger.debug(f"Scheduling meeting for {email} on {preferred_date} at {preferred_time}")
        if not self.service:
            return {"error": "Calendar service not available."}

        try:
            start_time = datetime.fromisoformat(
                f"{preferred_date}T{preferred_time}:00"
            )
            end_time = start_time + timedelta(hours=1)

            event = {
                "summary": "Aven Customer Support Call",
                "start": {"dateTime": start_time.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end_time.isoformat(), "timeZone": "UTC"},
                "attendees": [{"email": email}],
            }

            self.service.events().insert(
                calendarId="primary", body=event, sendUpdates="all"
            ).execute()
            return {
                "status": "success",
                "message": f"Meeting scheduled for {email} on {preferred_date} at {preferred_time}.",
            }
        except Exception as e:
            logger.error(f"Error scheduling meeting: {e}", exc_info=True)
            return {"error": str(e)}

    async def check_availability(self, date: str, time: str):
        logger.debug(f"Checking availability for {date} at {time}")
        if not self.service:
            return {"error": "Calendar service not available."}

        try:
            start_time = datetime.fromisoformat(f"{date}T{time}:00")
            end_time = start_time + timedelta(hours=1)

            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=start_time.isoformat() + "Z",
                    timeMax=end_time.isoformat() + "Z",
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])

            if not events:
                return {"available": True, "message": "The time slot is available."}
            else:
                return {
                    "available": False,
                    "message": "The time slot is not available.",
                }
        except Exception as e:
            logger.error(f"Error checking availability: {e}", exc_info=True)
            return {"error": str(e)}


def get_system_prompt():
    today_date = date.today().strftime("%A, %B %d, %Y")
    return f"""You are Aven's official customer support AI assistant, a friendly and professional expert on Aven's products and services. Your primary goal is to assist users with their questions about Aven.

**--- Context ---**
Today's date is {today_date}. Please use this as a reference for any date-related questions.

**--- Core Instructions & Guardrails ---**
1.  **STAY ON TOPIC**: Your knowledge is strictly limited to Aven. Politely refuse to answer any questions about other companies, general knowledge, or any topic not directly related to Aven. For off-topic questions, respond with: "I can only help with questions related to Aven. For other inquiries, you can contact our support team at support@aven.com."
2.  **DO NOT HALLUCINATE**: If you don't know the answer or if the tools don't provide the necessary information, do not make one up. Instead, say: "I couldn't find the information for that. For the most accurate details, please contact our support team at support@aven.com."
3.  **USE YOUR TOOLS**: You have several tools to help you. Use them intelligently based on the user's request.

**--- Tool Usage Guide ---**

*   **`search_aven_knowledge(query: str)`**:
    *   **USE THIS FIRST** for any questions about Aven's products (HELOC, credit cards), services, policies, rates, fees, application process, or company information.
    *   Example: If the user asks "what are your interest rates?", call `search_aven_knowledge(query="Aven interest rates")`.

*   **`search_web(query: str)`**:
    *   **USE THIS ONLY IF** `search_aven_knowledge` fails or doesn't provide a satisfactory answer, especially for recent news or topics that might not be in the knowledge base.

*   **`check_availability(date: str, time: str)` and `schedule_meeting(email: str, preferred_date: str, preferred_time: str)`**:
    *   **USE THESE** when the user explicitly asks to schedule, book, or check a time for a meeting or appointment.
    *   **Workflow:**
        1. When the user asks to schedule, first ask for their preferred date and time.
        2. Use `check_availability` to see if that slot is free.
        3. If the time is available, ask for their email address to send the calendar invite.
        4. **Confirm the email address** by repeating it back to them. For example: "Got it. Just to confirm, your email is example@email.com. Is that correct?"
        5. Once they confirm, call `schedule_meeting` with the details. The meeting topic will automatically be set to 'Aven Customer Support Call'.
        6. If their preferred time is not available, inform them and ask for an alternative time.
    *   Example: User says "I'd like to schedule a meeting." You should respond: "Certainly! What date and time would you like to schedule the meeting for?"

After using a tool, synthesize the information into a clear, concise, and friendly response for the user."""


async def list_assistants(vapi: Vapi):
    try:
        assistants = vapi.assistants.list()
        return assistants
    except Exception as e:
        logger.error(f"Error listing assistants: {e}", exc_info=True)
        return []


async def get_or_create_assistant(vapi: Vapi, tools: List[Dict[str, Any]]):
    ASSISTANT_NAME = "Aven Support AI (MVP)"
    today_date = date.today().strftime("%A, %B %d, %Y")

    # Get webhook URL
    webhook_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
    if webhook_url.startswith('http://'):
        webhook_url = webhook_url.replace('http://', 'https://')
    
    assistant_config = {
        "name": ASSISTANT_NAME,
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
                    "content": get_system_prompt()
                }
            ],
            "tools": tools,
        },
        "voice": {
            "provider": "11labs", 
            "voiceId": "pNInz6obpgDQGcFmaJgB"
        },
        "first_message": "Hello! Welcome to Aven support. I'm your AI assistant. How can I help you with our HELOC or credit card products today?",
        "end_call_message": "Thank you for contacting Aven support. Have a great day!",
        "server": {
            "url": f"{webhook_url}/vapi/webhook"
        }
    }

    try:
        logger.info(f"Searching for existing assistant named '{ASSISTANT_NAME}'...")
        assistants = await list_assistants(vapi)
        existing_assistant = None
        for assistant in assistants:
            if assistant.name == ASSISTANT_NAME:
                existing_assistant = assistant
                break

        if existing_assistant:
            logger.info(f"Found existing assistant with ID: {existing_assistant.id}. Updating it now.")
            logger.debug(f"Updating assistant with config: {assistant_config}")
            # Exclude 'name' from the update payload as it may not be allowed
            update_payload = {k: v for k, v in assistant_config.items() if k != 'name'}
            updated_assistant = vapi.assistants.update(
                id=existing_assistant.id, **update_payload
            )
            logger.info(f"Assistant {updated_assistant.id} updated successfully.")
            return updated_assistant
        else:
            logger.info("No existing assistant found. Creating a new one.")
            logger.debug(f"Creating assistant with config: {assistant_config}")
            new_assistant = vapi.assistants.create(**assistant_config)
            logger.info(f"Created new assistant with ID: {new_assistant.id}")
            return new_assistant

    except Exception as e:
        logger.error(f"Error getting or creating assistant: {e}", exc_info=True)
        return None 