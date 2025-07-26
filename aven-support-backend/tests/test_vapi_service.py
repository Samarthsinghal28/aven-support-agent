import os
import pytest
import unittest.mock as mock
import sys
import os.path
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapi_service import VapiService, MockCallResponse

class TestVapiService:
    @pytest.fixture
    def vapi_service(self):
        # Mock the Vapi client
        with mock.patch('vapi.Vapi') as mock_vapi:
            # Mock the OpenAI client
            with mock.patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the tools
                with mock.patch('vapi_service.RAGTool') as mock_rag_tool:
                    with mock.patch('vapi_service.SerperTool') as mock_serper_tool:
                        with mock.patch('vapi_service.CalendarTool') as mock_calendar_tool:
                            # Set up the mock tools
                            mock_rag_tool.return_value.schema = {"type": "function", "function": {"name": "search_aven_knowledge"}}
                            mock_serper_tool.return_value.schema = {"type": "function", "function": {"name": "search_web"}}
                            mock_calendar_tool.return_value.schedule_schema = {"type": "function", "function": {"name": "schedule_meeting"}}
                            mock_calendar_tool.return_value.availability_schema = {"type": "function", "function": {"name": "check_availability"}}
                            
                            # Mock the environment variables
                            with mock.patch.dict(os.environ, {"VAPI_API_KEY": "test_vapi_key", "OPENAI_API_KEY": "test_openai_key"}):
                                service = VapiService()
                                # Set the mocked objects
                                service.vapi = mock_vapi.return_value
                                service.openai_client = mock_openai.return_value
                                yield service
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_search_aven_knowledge(self, vapi_service):
        # Mock the RAGTool.use method
        vapi_service.rag_tool.use = mock.AsyncMock(return_value={"contexts": ["Test context"]})
        
        result = await vapi_service.handle_tool_call("search_aven_knowledge", {"query": "test query"})
        
        # Verify the result
        assert result == {"contexts": ["Test context"]}
        # Verify the tool was called with the right parameters
        vapi_service.rag_tool.use.assert_called_once_with(query="test query")
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_search_web(self, vapi_service):
        # Mock the SerperTool.use method
        vapi_service.serper_tool.use = mock.AsyncMock(return_value={"organic": [{"title": "Test result"}]})
        
        result = await vapi_service.handle_tool_call("search_web", {"query": "test query"})
        
        # Verify the result
        assert result == {"organic": [{"title": "Test result"}]}
        # Verify the tool was called with the right parameters
        vapi_service.serper_tool.use.assert_called_once_with(query="test query")
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_schedule_meeting(self, vapi_service):
        # Mock the CalendarTool.schedule method
        vapi_service.calendar_tool.schedule = mock.AsyncMock(return_value={"status": "success"})
        
        result = await vapi_service.handle_tool_call("schedule_meeting", {
            "email": "test@example.com",
            "preferred_date": "2023-12-31",
            "preferred_time": "14:30"
        })
        
        # Verify the result
        assert result == {"status": "success"}
        # Verify the tool was called with the right parameters
        vapi_service.calendar_tool.schedule.assert_called_once_with(
            email="test@example.com",
            preferred_date="2023-12-31",
            preferred_time="14:30"
        )
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_check_availability(self, vapi_service):
        # Mock the CalendarTool.check_availability method
        vapi_service.calendar_tool.check_availability = mock.AsyncMock(return_value={"available": True})
        
        result = await vapi_service.handle_tool_call("check_availability", {
            "date": "2023-12-31",
            "time": "14:30"
        })
        
        # Verify the result
        assert result == {"available": True}
        # Verify the tool was called with the right parameters
        vapi_service.calendar_tool.check_availability.assert_called_once_with(
            date="2023-12-31",
            time="14:30"
        )
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_unknown_tool(self, vapi_service):
        result = await vapi_service.handle_tool_call("unknown_tool", {})
        
        # Verify the error message
        assert "error" in result
        assert result["error"] == "Unknown tool"
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_exception(self, vapi_service):
        # Mock the RAGTool.use method to raise an exception
        vapi_service.rag_tool.use = mock.AsyncMock(side_effect=Exception("Test error"))
        
        result = await vapi_service.handle_tool_call("search_aven_knowledge", {"query": "test query"})
        
        # Verify the error message
        assert "error" in result
        assert result["error"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_cached(self, vapi_service):
        # Set a cached assistant ID
        vapi_service._cached_assistant_id = "test_assistant_id"
        
        result = await vapi_service.get_or_create_assistant()
        
        # Verify the cached ID is returned
        assert result == "test_assistant_id"
        # Verify no API calls were made
        vapi_service.vapi.assistants.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_new(self, vapi_service):
        # Mock the get_or_create_assistant function
        with mock.patch('vapi_service.get_or_create_assistant') as mock_get_or_create:
            mock_assistant = mock.MagicMock()
            mock_assistant.id = "new_assistant_id"
            mock_get_or_create.return_value = mock_assistant
            
            result = await vapi_service.get_or_create_assistant()
            
            # Verify the new ID is returned and cached
            assert result == "new_assistant_id"
            assert vapi_service._cached_assistant_id == "new_assistant_id"
            # Verify the function was called with the right parameters
            mock_get_or_create.assert_called_once_with(vapi_service.vapi, vapi_service.get_tools_schema())
    
    @pytest.mark.asyncio
    async def test_get_or_create_assistant_no_vapi(self, vapi_service):
        # Set vapi to None to simulate missing API key
        vapi_service.vapi = None
        
        result = await vapi_service.get_or_create_assistant()
        
        # Verify None is returned
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_tools_schema(self, vapi_service):
        # Set up the mock schemas
        vapi_service.rag_tool.schema = {"name": "search_aven_knowledge"}
        vapi_service.serper_tool.schema = {"name": "search_web"}
        vapi_service.calendar_tool.schedule_schema = {"name": "schedule_meeting"}
        vapi_service.calendar_tool.availability_schema = {"name": "check_availability"}
        
        result = vapi_service.get_tools_schema()
        
        # Verify all schemas are included
        assert len(result) == 4
        assert result[0] == {"name": "search_aven_knowledge"}
        assert result[1] == {"name": "search_web"}
        assert result[2] == {"name": "schedule_meeting"}
        assert result[3] == {"name": "check_availability"}
    
    @pytest.mark.asyncio
    async def test_process_chat_message_new_session(self, vapi_service):
        # Mock the system prompt
        with mock.patch('vapi_service.get_system_prompt', return_value="System prompt"):
            # Mock the OpenAI response
            mock_message = mock.MagicMock()
            mock_message.content = "Test response"
            mock_message.tool_calls = None
            
            mock_choice = mock.MagicMock()
            mock_choice.message = mock_message
            
            mock_response = mock.MagicMock()
            mock_response.choices = [mock_choice]
            
            vapi_service.openai_client.chat.completions.create = mock.AsyncMock(return_value=mock_response)
            
            result = await vapi_service.process_chat_message("Hello", "session123")
            
            # Verify the result
            assert result == "Test response"
            # Verify the session history was created
            assert "session123" in vapi_service.session_history
            assert len(vapi_service.session_history["session123"]) == 3  # System, user, assistant
            assert vapi_service.session_history["session123"][0]["role"] == "system"
            assert vapi_service.session_history["session123"][1]["role"] == "user"
            assert vapi_service.session_history["session123"][1]["content"] == "Hello"
            assert vapi_service.session_history["session123"][2]["role"] == "assistant"
            assert vapi_service.session_history["session123"][2]["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_process_chat_message_with_tool_calls(self, vapi_service):
        # Mock the system prompt
        with mock.patch('vapi_service.get_system_prompt', return_value="System prompt"):
            # Set up session history
            vapi_service.session_history["session123"] = [
                {"role": "system", "content": "System prompt"}
            ]
            
            # Mock the tool call
            mock_function = mock.MagicMock()
            mock_function.name = "search_aven_knowledge"
            mock_function.arguments = '{"query": "test query"}'
            
            mock_tool_call = mock.MagicMock()
            mock_tool_call.id = "tool123"
            mock_tool_call.function = mock_function
            
            # Mock the first OpenAI response with tool call
            mock_message1 = mock.MagicMock()
            mock_message1.content = None
            mock_message1.tool_calls = [mock_tool_call]
            
            mock_choice1 = mock.MagicMock()
            mock_choice1.message = mock_message1
            
            mock_response1 = mock.MagicMock()
            mock_response1.choices = [mock_choice1]
            
            # Mock the second OpenAI response with final answer
            mock_message2 = mock.MagicMock()
            mock_message2.content = "Final answer"
            mock_message2.tool_calls = None
            
            mock_choice2 = mock.MagicMock()
            mock_choice2.message = mock_message2
            
            mock_response2 = mock.MagicMock()
            mock_response2.choices = [mock_choice2]
            
            # Set up the mock to return different responses on consecutive calls
            vapi_service.openai_client.chat.completions.create = mock.AsyncMock(side_effect=[mock_response1, mock_response2])
            
            # Mock the tool call handler
            vapi_service.handle_tool_call = mock.AsyncMock(return_value={"result": "tool result"})
            
            result = await vapi_service.process_chat_message("Hello", "session123")
            
            # Verify the result
            assert result == "Final answer"
            # Verify the tool call was handled
            vapi_service.handle_tool_call.assert_called_once_with(
                function_name="search_aven_knowledge",
                parameters={"query": "test query"}
            )
    
    @pytest.mark.asyncio
    async def test_process_chat_message_exception(self, vapi_service):
        # Mock the system prompt
        with mock.patch('vapi_service.get_system_prompt', return_value="System prompt"):
            # Mock the OpenAI client to raise an exception
            vapi_service.openai_client.chat.completions.create = mock.AsyncMock(side_effect=Exception("Test error"))
            
            result = await vapi_service.process_chat_message("Hello", "session123")
            
            # Verify the error message
            assert "I'm experiencing technical difficulties" in result
    
    @pytest.mark.asyncio
    async def test_create_web_call_success(self, vapi_service):
        # Mock the get_or_create_assistant method
        vapi_service.get_or_create_assistant = mock.AsyncMock(return_value="test_assistant_id")
        
        result = await vapi_service.create_web_call()
        
        # Verify the result
        assert isinstance(result, MockCallResponse)
        assert result.id == "web_call_test_assistant_id"
        assert result.assistant_id == "test_assistant_id"
        assert result.type == "web"
        assert result.status == "ready"
    
    @pytest.mark.asyncio
    async def test_create_web_call_no_vapi(self, vapi_service):
        # Set vapi to None to simulate missing API key
        vapi_service.vapi = None
        
        with pytest.raises(Exception) as excinfo:
            await vapi_service.create_web_call()
        
        # Verify the error message
        assert "Vapi client not initialized" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_create_web_call_no_assistant(self, vapi_service):
        # Mock the get_or_create_assistant method to return None
        vapi_service.get_or_create_assistant = mock.AsyncMock(return_value=None)
        
        with pytest.raises(Exception) as excinfo:
            await vapi_service.create_web_call()
        
        # Verify the error message
        assert "Failed to get or create assistant" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_create_call_success(self, vapi_service):
        # Mock the get_or_create_assistant method
        vapi_service.get_or_create_assistant = mock.AsyncMock(return_value="test_assistant_id")
        # Mock the vapi.calls.create method
        vapi_service.vapi.calls.create = mock.MagicMock(return_value={"id": "call123"})
        
        result = await vapi_service.create_call("1234567890")
        
        # Verify the result
        assert result == {"id": "call123"}
        # Verify the method was called with the right parameters
        vapi_service.vapi.calls.create.assert_called_once_with(
            assistant_id="test_assistant_id",
            customer={"number": "1234567890"}
        )
    
    def test_get_call_status(self, vapi_service):
        # Mock the vapi.calls.get method
        vapi_service.vapi.calls.get = mock.MagicMock(return_value={"id": "call123", "status": "active"})
        
        result = vapi_service.get_call_status("call123")
        
        # Verify the result
        assert result == {"id": "call123", "status": "active"}
        # Verify the method was called with the right parameters
        vapi_service.vapi.calls.get.assert_called_once_with("call123")
    
    def test_end_call(self, vapi_service):
        # Mock the vapi.calls.delete method
        vapi_service.vapi.calls.delete = mock.MagicMock(return_value={"id": "call123", "status": "ended"})
        
        result = vapi_service.end_call("call123")
        
        # Verify the result
        assert result == {"id": "call123", "status": "ended"}
        # Verify the method was called with the right parameters
        vapi_service.vapi.calls.delete.assert_called_once_with("call123")
    
    def test_list_calls(self, vapi_service):
        # Mock the vapi.calls.list method
        vapi_service.vapi.calls.list = mock.MagicMock(return_value={"calls": [{"id": "call123"}]})
        
        result = vapi_service.list_calls(limit=10)
        
        # Verify the result
        assert result == {"calls": [{"id": "call123"}]}
        # Verify the method was called with the right parameters
        vapi_service.vapi.calls.list.assert_called_once_with(limit=10)

class TestMockCallResponse:
    def test_init(self):
        call_data = {
            "id": "call123",
            "assistant_id": "assistant123",
            "type": "web",
            "status": "ready",
            "message": "Test message",
            "extra_field": "extra value"
        }
        
        response = MockCallResponse(call_data)
        
        # Verify all fields are set correctly
        assert response.id == "call123"
        assert response.assistant_id == "assistant123"
        assert response.type == "web"
        assert response.status == "ready"
        assert response.message == "Test message"
        assert response.extra_field == "extra value" 