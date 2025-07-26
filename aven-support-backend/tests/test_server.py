import pytest
from fastapi.testclient import TestClient
import unittest.mock as mock
import sys
import os.path
import json
import time

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server
from server import app, vapi_service

client = TestClient(app)

def test_root():
    """Test the root endpoint returns a healthy status"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Aven Support AI is running"}

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    
    # Verify the response structure
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert isinstance(data["agent_available"], bool)
    assert isinstance(data["vapi_available"], bool)
    assert isinstance(data["active_sessions"], int)
    assert isinstance(data["active_vapi_calls"], int)
    assert isinstance(data["timestamp"], float)

@pytest.mark.asyncio
async def test_chat_handler_with_message():
    """Test the chat endpoint with a message"""
    # Mock the vapi_service.process_chat_message method
    with mock.patch.object(vapi_service, 'process_chat_message', return_value="Test response"):
        # Mock the vapi_service.get_or_create_assistant method
        with mock.patch.object(vapi_service, 'get_or_create_assistant', return_value="test_assistant_id"):
            # Create a test payload
            payload = {
                "message": "Hello",
                "session_id": "test_session"
            }
            
            # Send the request
            response = client.post("/chat", json=payload)
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"
            assert data["assistantId"] == "test_assistant_id"
            assert data["session_id"] == "test_session"
            assert isinstance(data["response_time"], float)
            
            # Verify the service methods were called correctly
            vapi_service.process_chat_message.assert_called_once_with("Hello", "test_session")
            vapi_service.get_or_create_assistant.assert_called_once()

@pytest.mark.asyncio
async def test_chat_handler_no_message():
    """Test the chat endpoint without a message (just getting assistant ID)"""
    # Mock the vapi_service methods
    original_process_chat = vapi_service.process_chat_message
    original_get_assistant = vapi_service.get_or_create_assistant
    
    # Create mock functions
    process_chat_mock = mock.AsyncMock(return_value="Test response")
    get_assistant_mock = mock.AsyncMock(return_value="test_assistant_id")
    
    try:
        # Replace with mocks
        vapi_service.process_chat_message = process_chat_mock
        vapi_service.get_or_create_assistant = get_assistant_mock
        
        # Create a test payload
        payload = {
            "session_id": "test_session"
        }
        
        # Send the request
        response = client.post("/chat", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == ""  # No message, no response
        assert data["assistantId"] == "test_assistant_id"
        assert data["session_id"] == "test_session"
        
        # Verify the service methods were called correctly
        assert process_chat_mock.call_count == 0
        assert get_assistant_mock.call_count == 1
    finally:
        # Restore original methods
        vapi_service.process_chat_message = original_process_chat
        vapi_service.get_or_create_assistant = original_get_assistant

@pytest.mark.asyncio
async def test_chat_handler_vapi_error():
    """Test the chat endpoint when Vapi is unavailable"""
    # Mock the vapi_service.process_chat_message method
    with mock.patch.object(vapi_service, 'process_chat_message', return_value="Test response"):
        # Mock the vapi_service.get_or_create_assistant method to return None
        with mock.patch.object(vapi_service, 'get_or_create_assistant', return_value=None):
            # Create a test payload
            payload = {
                "message": "Hello",
                "session_id": "test_session"
            }
            
            # Send the request
            response = client.post("/chat", json=payload)
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Test response"
            assert data["assistantId"] is None
            assert data["session_id"] == "test_session"

def test_vapi_webhook_tool_calls():
    """Test the Vapi webhook endpoint for tool calls"""
    # Mock the vapi_service.handle_tool_call method
    with mock.patch.object(vapi_service, 'handle_tool_call', return_value={"result": "test result"}):
        # Create a test payload
        payload = {
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "tool123",
                        "function": {
                            "name": "search_aven_knowledge",
                            "arguments": '{"query": "test query"}'
                        }
                    }
                ]
            }
        }
        
        # Send the request
        response = client.post("/vapi/webhook", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        assert response.json() == {
            "results": [
                {
                    "toolCallId": "tool123",
                    "result": {"result": "test result"}
                }
            ]
        }
        
        # Verify the service method was called correctly
        vapi_service.handle_tool_call.assert_called_once_with(
            "search_aven_knowledge", 
            {"query": "test query"}
        )

def test_vapi_webhook_tool_calls_dict_arguments():
    """Test the Vapi webhook endpoint with arguments as a dict instead of a string"""
    # Mock the vapi_service.handle_tool_call method
    with mock.patch.object(vapi_service, 'handle_tool_call', return_value={"result": "test result"}):
        # Create a test payload with arguments as a dict
        payload = {
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "tool123",
                        "function": {
                            "name": "search_aven_knowledge",
                            "arguments": {"query": "test query"}
                        }
                    }
                ]
            }
        }
        
        # Send the request
        response = client.post("/vapi/webhook", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        assert response.json() == {
            "results": [
                {
                    "toolCallId": "tool123",
                    "result": {"result": "test result"}
                }
            ]
        }
        
        # Verify the service method was called correctly
        vapi_service.handle_tool_call.assert_called_once_with(
            "search_aven_knowledge", 
            {"query": "test query"}
        )

def test_vapi_webhook_unknown_message_type():
    """Test the Vapi webhook endpoint with an unknown message type"""
    payload = {
        "message": {
            "type": "unknown",
            "content": "test content"
        }
    }
    
    response = client.post("/vapi/webhook", json=payload)
    
    # Should still return 200 to prevent Vapi from retrying
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_vapi_webhook_multiple_tool_calls():
    """Test the Vapi webhook endpoint with multiple tool calls"""
    # Mock the vapi_service.handle_tool_call method with different returns
    async def mock_handle_tool_call(name, params):
        if name == "search_aven_knowledge":
            return {"knowledge": "test knowledge"}
        elif name == "search_web":
            return {"web_results": "test web results"}
        else:
            return {"error": "Unknown tool"}
    
    with mock.patch.object(vapi_service, 'handle_tool_call', side_effect=mock_handle_tool_call):
        # Create a test payload with multiple tool calls
        payload = {
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "tool1",
                        "function": {
                            "name": "search_aven_knowledge",
                            "arguments": '{"query": "knowledge query"}'
                        }
                    },
                    {
                        "id": "tool2",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query": "web query"}'
                        }
                    }
                ]
            }
        }
        
        # Send the request
        response = client.post("/vapi/webhook", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        
        # Check first tool call result
        assert results[0]["toolCallId"] == "tool1"
        assert results[0]["result"] == {"knowledge": "test knowledge"}
        
        # Check second tool call result
        assert results[1]["toolCallId"] == "tool2"
        assert results[1]["result"] == {"web_results": "test web results"}

def test_vapi_webhook_legacy_function_call():
    """Test the Vapi webhook endpoint with the legacy function-call format"""
    # Mock the vapi_service.handle_tool_call method
    with mock.patch.object(vapi_service, 'handle_tool_call', return_value={"result": "test result"}):
        # Create a test payload with the legacy format
        payload = {
            "message": {
                "type": "function-call",
                "functionCall": {
                    "name": "search_aven_knowledge",
                    "parameters": {"query": "test query"}
                }
            }
        }
        
        # Send the request
        response = client.post("/vapi/webhook", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        assert response.json() == {"result": {"result": "test result"}}
        
        # Verify the service method was called correctly
        vapi_service.handle_tool_call.assert_called_once_with(
            "search_aven_knowledge", 
            {"query": "test query"}
        )

def test_vapi_webhook_error():
    """Test the Vapi webhook endpoint when an error occurs"""
    # Mock the vapi_service.handle_tool_call method to raise an exception
    with mock.patch.object(vapi_service, 'handle_tool_call', side_effect=Exception("Test error")):
        # Create a test payload
        payload = {
            "message": {
                "type": "tool-calls",
                "toolCalls": [
                    {
                        "id": "tool123",
                        "function": {
                            "name": "search_aven_knowledge",
                            "arguments": '{"query": "test query"}'
                        }
                    }
                ]
            }
        }
        
        # Send the request
        response = client.post("/vapi/webhook", json=payload)
        
        # Should still return 200 to prevent Vapi from retrying
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Test error" in data["message"]

@pytest.mark.asyncio
async def test_create_vapi_assistant_success():
    """Test the create Vapi assistant endpoint"""
    # Mock the vapi_service.get_or_create_assistant method
    with mock.patch.object(vapi_service, 'get_or_create_assistant', return_value="test_assistant_id"):
        response = client.post("/vapi/assistant")
        
        # Verify the response
        assert response.status_code == 200
        assert response.json() == {"assistant_id": "test_assistant_id"}
        
        # Verify the service method was called
        vapi_service.get_or_create_assistant.assert_called_once()

@pytest.mark.asyncio
async def test_create_vapi_assistant_failure():
    """Test the create Vapi assistant endpoint when it fails"""
    # Create a custom mock that returns None and can be patched
    async def mock_get_or_create_assistant():
        return None
    
    # Mock the vapi_service.get_or_create_assistant method
    with mock.patch.object(vapi_service, 'get_or_create_assistant', side_effect=mock_get_or_create_assistant):
        # Mock the logger to prevent actual logging during the test
        with mock.patch('server.logger'):
            response = client.post("/vapi/assistant")
            
            # Verify the response
            assert response.status_code == 503
            assert "Service Unavailable" in response.json()["detail"]

@pytest.mark.asyncio
async def test_create_vapi_assistant_error():
    """Test the create Vapi assistant endpoint when an error occurs"""
    # Mock the vapi_service.get_or_create_assistant method to raise an exception
    with mock.patch.object(vapi_service, 'get_or_create_assistant', side_effect=Exception("Test error")):
        response = client.post("/vapi/assistant")
        
        # Verify the response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_create_vapi_web_call():
    """Test the create Vapi web call endpoint"""
    # Mock the vapi_service.create_web_call method
    mock_call = mock.MagicMock()
    mock_call.__dict__ = {
        "id": "call123",
        "assistant_id": "assistant123",
        "type": "web",
        "status": "ready"
    }
    
    with mock.patch.object(vapi_service, 'create_web_call', return_value=mock_call):
        # Create a test payload
        payload = {"type": "web"}
        
        response = client.post("/vapi/call", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["call_id"] == "call123"
        assert data["call_response"]["id"] == "call123"
        assert data["call_response"]["assistant_id"] == "assistant123"
        
        # Verify the service method was called
        vapi_service.create_web_call.assert_called_once()

@pytest.mark.asyncio
async def test_create_vapi_phone_call():
    """Test the create Vapi phone call endpoint"""
    # Create a mock response with the expected structure
    mock_response = mock.MagicMock()
    mock_response.id = "call123"
    mock_response.__dict__ = {"id": "call123"}
    
    # Mock the vapi_service.create_call method
    with mock.patch.object(vapi_service, 'create_call', return_value=mock_response):
        # Create a test payload
        payload = {"type": "phone", "phone_number": "1234567890"}
        
        response = client.post("/vapi/call", json=payload)
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["call_id"] == "call123"
        
        # Verify the service method was called correctly
        vapi_service.create_call.assert_called_once_with("1234567890")

@pytest.mark.asyncio
async def test_create_vapi_call_error():
    """Test the create Vapi call endpoint when an error occurs"""
    # Mock the vapi_service.create_web_call method to raise an exception
    with mock.patch.object(vapi_service, 'create_web_call', side_effect=Exception("Test error")):
        # Create a test payload
        payload = {"type": "web"}
        
        response = client.post("/vapi/call", json=payload)
        
        # Verify the response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

def test_get_vapi_call_status():
    """Test the get Vapi call status endpoint"""
    # Mock the vapi_service.get_call_status method
    with mock.patch.object(vapi_service, 'get_call_status', return_value={"id": "call123", "status": "active"}):
        response = client.get("/vapi/call/call123")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["call_id"] == "call123"
        assert data["status"]["id"] == "call123"
        assert data["status"]["status"] == "active"
        
        # Verify the service method was called correctly
        vapi_service.get_call_status.assert_called_once_with("call123")

def test_get_vapi_call_status_error():
    """Test the get Vapi call status endpoint when an error occurs"""
    # Mock the vapi_service.get_call_status method to raise an exception
    with mock.patch.object(vapi_service, 'get_call_status', side_effect=Exception("Test error")):
        response = client.get("/vapi/call/call123")
        
        # Verify the response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

def test_end_vapi_call():
    """Test the end Vapi call endpoint"""
    # Mock the vapi_service.end_call method
    with mock.patch.object(vapi_service, 'end_call', return_value={"id": "call123", "status": "ended"}):
        # Mock the active_vapi_calls dict
        original_active_calls = server.active_vapi_calls
        server.active_vapi_calls = {"call123": {"status": "active"}}
        
        try:
            response = client.post("/vapi/call/call123/end")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["call_id"] == "call123"
            assert data["result"]["id"] == "call123"
            assert data["result"]["status"] == "ended"
            
            # Verify the service method was called correctly
            vapi_service.end_call.assert_called_once_with("call123")
            
            # Verify the active_vapi_calls dict was updated
            assert server.active_vapi_calls["call123"]["status"] == "ended"
            assert "ended_at" in server.active_vapi_calls["call123"]
        finally:
            # Restore the original dict
            server.active_vapi_calls = original_active_calls

def test_end_vapi_call_error():
    """Test the end Vapi call endpoint when an error occurs"""
    # Mock the vapi_service.end_call method to raise an exception
    with mock.patch.object(vapi_service, 'end_call', side_effect=Exception("Test error")):
        response = client.post("/vapi/call/call123/end")
        
        # Verify the response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

def test_list_active_vapi_calls():
    """Test the list active Vapi calls endpoint"""
    # Mock the vapi_service.list_calls method
    with mock.patch.object(vapi_service, 'list_calls', return_value={"calls": [{"id": "call123"}]}):
        # Mock the active_vapi_calls dict
        original_active_calls = server.active_vapi_calls
        server.active_vapi_calls = {"call123": {"status": "active"}}
        
        try:
            response = client.get("/vapi/calls")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["active_calls"] == {"call123": {"status": "active"}}
            assert data["recent_vapi_calls"]["calls"][0]["id"] == "call123"
            
            # Verify the service method was called correctly
            vapi_service.list_calls.assert_called_once_with(limit=20)
        finally:
            # Restore the original dict
            server.active_vapi_calls = original_active_calls

def test_list_active_vapi_calls_error():
    """Test the list active Vapi calls endpoint when an error occurs"""
    # Mock the vapi_service.list_calls method to raise an exception
    with mock.patch.object(vapi_service, 'list_calls', side_effect=Exception("Test error")):
        # Mock the active_vapi_calls dict
        original_active_calls = server.active_vapi_calls
        server.active_vapi_calls = {"call123": {"status": "active"}}
        
        try:
            response = client.get("/vapi/calls")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["active_calls"] == {"call123": {"status": "active"}}
            assert data["recent_vapi_calls"] == []
            assert data["error"] == "Test error"
        finally:
            # Restore the original dict
            server.active_vapi_calls = original_active_calls 