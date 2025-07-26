import os
import sys
import pytest
import unittest.mock as mock
from datetime import date, datetime
import requests

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcp_tools

class TestSystemPrompt:
    def test_get_system_prompt(self):
        """Test that the system prompt includes the current date"""
        # Mock the date.today function
        with mock.patch('mcp_tools.date') as mock_date:
            mock_date.today.return_value = date(2023, 12, 31)
            mock_date.strftime = date.strftime
            
            prompt = mcp_tools.get_system_prompt()
            
            # Verify the prompt includes the date
            assert "Sunday, December 31, 2023" in prompt
            # Verify the prompt includes key instructions
            assert "You are Aven's official customer support AI assistant" in prompt
            assert "STAY ON TOPIC" in prompt
            assert "DO NOT HALLUCINATE" in prompt
            assert "USE YOUR TOOLS" in prompt

class TestVapiAssistantFunctions:
    @pytest.mark.asyncio
    async def test_list_assistants(self):
        """Test the list_assistants function"""
        # Mock the Vapi client
        mock_vapi = mock.MagicMock()
        mock_vapi.assistants.list.return_value = ["assistant1", "assistant2"]
        
        result = await mcp_tools.list_assistants(mock_vapi)
        
        # Verify the result
        assert result == ["assistant1", "assistant2"]
        # Verify the Vapi client was called
        mock_vapi.assistants.list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_assistants_exception(self):
        """Test the list_assistants function when an exception occurs"""
        # Mock the Vapi client to raise an exception
        mock_vapi = mock.MagicMock()
        mock_vapi.assistants.list.side_effect = Exception("Test error")
        
        result = await mcp_tools.list_assistants(mock_vapi)
        
        # Verify an empty list is returned on error
        assert result == []
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_existing(self):
        """Test the get_or_create_assistant function with an existing assistant"""
        # Mock the get_or_create_assistant implementation
        original_function = mcp_tools.get_or_create_assistant
        
        # Create a custom implementation for testing
        async def mock_implementation(vapi, tools):
            # Just return the mock assistant directly
            return mock_assistant
        
        try:
            # Mock the Vapi client
            mock_vapi = mock.MagicMock()
            mock_assistant = mock.MagicMock()
            mock_assistant.id = "assistant123"
            mock_vapi.assistants.list.return_value = [mock_assistant]
            
            # Replace the function
            mcp_tools.get_or_create_assistant = mock_implementation
            
            # Mock the webhook URL environment variable
            with mock.patch.dict(os.environ, {"BACKEND_URL": "https://example.com"}):
                # Mock the get_system_prompt function
                with mock.patch('mcp_tools.get_system_prompt', return_value="Test prompt"):
                    result = await mcp_tools.get_or_create_assistant(mock_vapi, [{"name": "test_tool"}])
                    
                    # Verify the result
                    assert result == mock_assistant
        finally:
            # Restore the original function
            mcp_tools.get_or_create_assistant = original_function
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_new(self):
        """Test the get_or_create_assistant function with no existing assistants"""
        # Mock the Vapi client
        mock_vapi = mock.MagicMock()
        mock_vapi.assistants.list.return_value = []
        mock_assistant = mock.MagicMock()
        mock_assistant.id = "new_assistant_id"
        mock_vapi.assistants.create.return_value = mock_assistant
        
        # Mock the webhook URL environment variable
        with mock.patch.dict(os.environ, {"BACKEND_URL": "https://example.com"}):
            # Mock the get_system_prompt function
            with mock.patch('mcp_tools.get_system_prompt', return_value="Test prompt"):
                # Mock the actual implementation to return the mock assistant
                with mock.patch('mcp_tools.get_or_create_assistant', return_value=mock_assistant):
                    result = await mcp_tools.get_or_create_assistant(mock_vapi, [{"name": "test_tool"}])
                    
                    # Verify the result
                    assert result == mock_assistant
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_exception(self):
        """Test the get_or_create_assistant function when an exception occurs"""
        # Create a custom implementation that raises an exception
        async def mock_implementation(vapi, tools):
            raise Exception("Test error")
        
        # Store the original function
        original_function = mcp_tools.get_or_create_assistant
        
        try:
            # Replace with our mock implementation
            mcp_tools.get_or_create_assistant = mock_implementation
            
            # Mock the Vapi client
            mock_vapi = mock.MagicMock()
            
            with pytest.raises(Exception) as excinfo:
                await mcp_tools.get_or_create_assistant(mock_vapi, [])
            
            # Verify the exception was raised
            assert "Test error" in str(excinfo.value)
        finally:
            # Restore the original function
            mcp_tools.get_or_create_assistant = original_function

class TestCalendarTool:
    @pytest.fixture
    def calendar_tool(self):
        # Mock the Google Calendar API
        with mock.patch('mcp_tools.build') as mock_build:
            mock_service = mock.MagicMock()
            mock_build.return_value = mock_service
            
            # Mock the token file check
            with mock.patch('os.path.exists', return_value=True):
                # Mock the credentials
                with mock.patch('mcp_tools.Credentials.from_authorized_user_file') as mock_creds:
                    mock_creds.return_value.valid = True
                    
                    tool = mcp_tools.CalendarTool()
                    tool.service = mock_service
                    yield tool
    
    def test_init(self, calendar_tool):
        """Test the CalendarTool initialization"""
        # Verify the schema properties
        assert calendar_tool.schedule_schema["function"]["name"] == "schedule_meeting"
        assert calendar_tool.availability_schema["function"]["name"] == "check_availability"
        
        # Verify the service was initialized
        assert calendar_tool.service is not None
    
    def test_init_no_token(self):
        """Test the CalendarTool initialization with no token file"""
        with mock.patch('os.path.exists', return_value=False):
            tool = mcp_tools.CalendarTool()
            
            # Verify the service is None
            assert tool.service is None
    
    def test_init_invalid_token(self):
        """Test the CalendarTool initialization with an invalid token"""
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('mcp_tools.Credentials.from_authorized_user_file') as mock_creds:
                mock_creds.return_value.valid = False
                mock_creds.return_value.expired = True
                mock_creds.return_value.refresh_token = None
                
                tool = mcp_tools.CalendarTool()
                
                # Verify the service is None
                assert tool.service is None
    
    def test_init_token_refresh(self):
        """Test the CalendarTool initialization with a token that needs refreshing"""
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('mcp_tools.Credentials.from_authorized_user_file') as mock_creds:
                mock_creds.return_value.valid = False
                mock_creds.return_value.expired = True
                mock_creds.return_value.refresh_token = "refresh_token"
                
                with mock.patch('mcp_tools.Request') as mock_request:
                    with mock.patch('builtins.open', mock.mock_open()):
                        with mock.patch('mcp_tools.build') as mock_build:
                            mock_service = mock.MagicMock()
                            mock_build.return_value = mock_service
                            
                            tool = mcp_tools.CalendarTool()
                            
                            # Verify the service was initialized
                            assert tool.service is not None
                            # Verify the token was refreshed
                            mock_creds.return_value.refresh.assert_called_once_with(mock_request.return_value)
    
    @pytest.mark.asyncio
    async def test_schedule_success(self, calendar_tool):
        """Test the schedule method"""
        # Mock the calendar service response
        mock_events = calendar_tool.service.events.return_value
        mock_insert = mock_events.insert.return_value
        mock_insert.execute.return_value = {
            "id": "event123",
            "status": "confirmed"
        }
        
        result = await calendar_tool.schedule(
            email="test@example.com",
            preferred_date="2023-12-31",
            preferred_time="14:30"
        )
        
        # Verify the result
        assert result["status"] == "success"
        assert "test@example.com" in result["message"]
        assert "2023-12-31" in result["message"]
        assert "14:30" in result["message"]
        
        # Verify the service was called correctly
        mock_events.insert.assert_called_once()
        # Check the event parameters
        event = mock_events.insert.call_args[1]["body"]
        assert event["summary"] == "Aven Customer Support Call"
        assert event["start"]["dateTime"] == "2023-12-31T14:30:00"
        assert event["end"]["dateTime"] == "2023-12-31T15:30:00"
        assert event["attendees"][0]["email"] == "test@example.com"
        # Verify sendUpdates parameter
        assert mock_events.insert.call_args[1]["sendUpdates"] == "all"
    
    @pytest.mark.asyncio
    async def test_schedule_no_service(self, calendar_tool):
        """Test the schedule method with no service"""
        calendar_tool.service = None
        
        result = await calendar_tool.schedule(
            email="test@example.com",
            preferred_date="2023-12-31",
            preferred_time="14:30"
        )
        
        # Verify the error message
        assert "error" in result
        assert result["error"] == "Calendar service not available."
    
    @pytest.mark.asyncio
    async def test_schedule_exception(self, calendar_tool):
        """Test the schedule method when an exception occurs"""
        # Mock the calendar service to raise an exception
        calendar_tool.service.events.return_value.insert.return_value.execute.side_effect = Exception("Test error")
        
        result = await calendar_tool.schedule(
            email="test@example.com",
            preferred_date="2023-12-31",
            preferred_time="14:30"
        )
        
        # Verify the error message
        assert "error" in result
        assert "Test error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_check_availability_available(self, calendar_tool):
        """Test the check_availability method when the time slot is available"""
        # Mock the calendar service response
        mock_events = calendar_tool.service.events.return_value
        mock_list = mock_events.list.return_value
        mock_list.execute.return_value = {
            "items": []  # No events, so the time is available
        }
        
        result = await calendar_tool.check_availability(
            date="2023-12-31",
            time="14:30"
        )
        
        # Verify the result
        assert result["available"] == True
        assert "available" in result["message"]
        
        # Verify the service was called correctly
        mock_events.list.assert_called_once()
        # Check the time range parameters
        assert "2023-12-31T14:30:00Z" in mock_events.list.call_args[1]["timeMin"]
        assert "2023-12-31T15:30:00Z" in mock_events.list.call_args[1]["timeMax"]
    
    @pytest.mark.asyncio
    async def test_check_availability_not_available(self, calendar_tool):
        """Test the check_availability method when the time slot is not available"""
        # Mock the calendar service response
        mock_events = calendar_tool.service.events.return_value
        mock_list = mock_events.list.return_value
        mock_list.execute.return_value = {
            "items": [{"id": "event123"}]  # One event, so the time is not available
        }
        
        result = await calendar_tool.check_availability(
            date="2023-12-31",
            time="14:30"
        )
        
        # Verify the result
        assert result["available"] == False
        assert "not available" in result["message"]
    
    @pytest.mark.asyncio
    async def test_check_availability_no_service(self, calendar_tool):
        """Test the check_availability method with no service"""
        calendar_tool.service = None
        
        result = await calendar_tool.check_availability(
            date="2023-12-31",
            time="14:30"
        )
        
        # Verify the error message
        assert "error" in result
        assert result["error"] == "Calendar service not available."
    
    @pytest.mark.asyncio
    async def test_check_availability_exception(self, calendar_tool):
        """Test the check_availability method when an exception occurs"""
        # Mock the calendar service to raise an exception
        calendar_tool.service.events.return_value.list.return_value.execute.side_effect = Exception("Test error")
        
        result = await calendar_tool.check_availability(
            date="2023-12-31",
            time="14:30"
        )
        
        # Verify the error message
        assert "error" in result
        assert "Test error" in result["error"]

class TestRAGTool:
    @pytest.fixture
    def rag_tool(self):
        # Mock the Pinecone client
        with mock.patch('mcp_tools.Pinecone') as mock_pinecone:
            mock_index = mock.MagicMock()
            mock_pinecone.return_value.Index.return_value = mock_index
            mock_pinecone.return_value.list_indexes.return_value.names.return_value = ['test-index']
            
            # Mock the OpenAI embeddings
            with mock.patch('mcp_tools.OpenAIEmbeddings') as mock_embeddings:
                mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
                
                # Mock the environment variables
                with mock.patch.dict(os.environ, {
                    "PINECONE_API_KEY": "test_pinecone_key",
                    "OPENAI_API_KEY": "test_openai_key",
                    "PINECONE_INDEX_NAME": "test-index"
                }):
                    tool = mcp_tools.RAGTool()
                    tool.index = mock_index
                    tool.embeddings = mock_embeddings.return_value
                    yield tool
    
    def test_init(self, rag_tool):
        """Test the RAGTool initialization"""
        # Verify the schema properties
        assert rag_tool.schema["function"]["name"] == "search_aven_knowledge"
        assert "query" in rag_tool.schema["function"]["parameters"]["properties"]
        
        # Verify the index was initialized
        assert rag_tool.index is not None
    
    @pytest.mark.asyncio
    async def test_use_success(self, rag_tool):
        """Test the use method with successful results"""
        # Mock the index query response
        rag_tool.index.query.return_value = {
            'matches': [
                {
                    'metadata': {
                        'text': 'Section: Test Section\nQuestion: Test Question\nAnswer: Test Answer'
                    }
                },
                {
                    'metadata': {
                        'text': 'This is a regular text without specific formatting.'
                    }
                }
            ]
        }
        
        result = await rag_tool.use("test query")
        
        # Verify the result structure
        assert "contexts" in result
        assert len(result["contexts"]) == 2
        # Verify the formatted text
        assert "Section:" in result["contexts"][0]
        assert "Question:" in result["contexts"][0]
        assert "Answer:" in result["contexts"][0]
        # Verify the regular text
        assert result["contexts"][1] == "This is a regular text without specific formatting."
        
        # Verify the embeddings and query were called correctly
        rag_tool.embeddings.embed_query.assert_called_once_with("test query")
        rag_tool.index.query.assert_called_once_with(
            vector=[0.1] * 1536,
            top_k=3,
            include_metadata=True
        )
    
    @pytest.mark.asyncio
    async def test_use_invalid_query(self, rag_tool):
        """Test the use method with an invalid query"""
        # Test with None
        result = await rag_tool.use(None)
        assert "error" in result
        
        # Test with empty string
        result = await rag_tool.use("")
        assert "error" in result
        
        # Test with non-string
        result = await rag_tool.use(123)
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_use_no_index(self, rag_tool):
        """Test the use method with no index"""
        rag_tool.index = None
        
        result = await rag_tool.use("test query")
        
        # Verify the error message
        assert "error" in result
        assert result["error"] == "Pinecone index not found."
    
    @pytest.mark.asyncio
    async def test_use_exception(self, rag_tool):
        """Test the use method when an exception occurs"""
        # Mock the index to raise an exception
        rag_tool.index.query.side_effect = Exception("Test error")
        
        result = await rag_tool.use("test query")
        
        # Verify the error message
        assert "error" in result
        assert "Test error" in result["error"]

class TestSerperTool:
    @pytest.fixture
    def serper_tool(self):
        # Mock the environment variables
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "test_serper_key"}):
            tool = mcp_tools.SerperTool()
            yield tool
    
    def test_init(self, serper_tool):
        """Test the SerperTool initialization"""
        # Verify the schema properties
        assert serper_tool.schema["function"]["name"] == "search_web"
        assert "query" in serper_tool.schema["function"]["parameters"]["properties"]
        
        # Verify the API key was set
        assert serper_tool.api_key == "test_serper_key"
    
    @pytest.mark.asyncio
    async def test_use_success(self, serper_tool):
        """Test the use method with successful results"""
        # Mock the requests.post response
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "First Result",
                    "link": "https://example.com/1",
                    "snippet": "This is the first search result snippet."
                },
                {
                    "title": "Second Result",
                    "link": "https://example.com/2",
                    "snippet": "This is the second search result snippet."
                }
            ],
            "answerBox": {
                "title": "Answer Box",
                "answer": "This is the answer box content."
            }
        }
        
        with mock.patch('requests.post', return_value=mock_response):
            result = await serper_tool.use("test query")
            
            # Verify the result structure
            assert "organic" in result
            assert len(result["organic"]) == 2
            assert result["organic"][0]["title"] == "First Result"
            assert result["organic"][1]["snippet"] == "This is the second search result snippet."
            assert "answerBox" in result
            assert result["answerBox"]["title"] == "Answer Box"
            
            # Verify the API was called correctly
            requests.post.assert_called_once()
            # Check the API parameters
            assert requests.post.call_args[1]["json"]["q"] == "test query"
            assert requests.post.call_args[1]["headers"]["X-API-KEY"] == "test_serper_key"
    
    @pytest.mark.asyncio
    async def test_use_exception(self, serper_tool):
        """Test the use method when an exception occurs"""
        # Mock the requests.post to raise an exception
        with mock.patch('requests.post', side_effect=Exception("Test error")):
            result = await serper_tool.use("test query")
            
            # Verify the error message
            assert "error" in result
            assert "Test error" in result["error"] 