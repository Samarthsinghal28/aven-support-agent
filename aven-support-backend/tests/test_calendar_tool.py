import os
import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import sys
import os.path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_tools import CalendarTool

class TestCalendarTool:
    @pytest.fixture
    def calendar_tool(self):
        with mock.patch('mcp_tools.build') as mock_build:
            mock_service = mock.MagicMock()
            mock_build.return_value = mock_service
            
            with mock.patch.dict(os.environ, {"GOOGLE_CALENDAR_CREDENTIALS_PATH": "test_path"}):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
                        tool = CalendarTool()
                        tool.service = mock_service
                        yield tool
    
    @pytest.mark.asyncio
    async def test_schedule_success(self, calendar_tool):
        # Mock the calendar service response
        mock_events = calendar_tool.service.events()
        mock_insert = mock_events.insert.return_value
        mock_insert.execute.return_value = {
            "id": "event123",
            "status": "confirmed"
        }
        
        # Test parameters
        email = "test@example.com"
        preferred_date = "2023-12-31"
        preferred_time = "14:30"
        
        result = await calendar_tool.schedule(email, preferred_date, preferred_time)
        
        # Verify the result
        assert result["status"] == "success"
        assert "Meeting scheduled for" in result["message"]
        assert email in result["message"]
        assert preferred_date in result["message"]
        assert preferred_time in result["message"]
        
        # Verify the service was called correctly
        mock_events.insert.assert_called_once_with(
            calendarId='primary',
            body={
                'summary': 'Aven Customer Support Call',
                'start': {'dateTime': '2023-12-31T14:30:00', 'timeZone': 'UTC'},
                'end': {'dateTime': '2023-12-31T15:30:00', 'timeZone': 'UTC'},
                'attendees': [{'email': email}],
            },
            sendUpdates="all"
        )
        mock_insert.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_schedule_no_service(self, calendar_tool):
        # Set service to None to simulate missing service
        calendar_tool.service = None
        
        result = await calendar_tool.schedule("test@example.com", "2023-12-31", "14:30")
        
        # Verify error message is returned
        assert "error" in result
        assert result["error"] == "Calendar service not available."
    
    @pytest.mark.asyncio
    async def test_schedule_exception(self, calendar_tool):
        # Make the service raise an exception
        calendar_tool.service.events().insert().execute.side_effect = Exception("Test error")
        
        result = await calendar_tool.schedule("test@example.com", "2023-12-31", "14:30")
        
        # Verify error message is returned
        assert "error" in result
        assert "Test error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_check_availability_available(self, calendar_tool):
        # Mock the calendar service to return no events (available)
        calendar_tool.service.events().list().execute.return_value = {
            "items": []
        }
        
        result = await calendar_tool.check_availability("2023-12-31", "14:30")
        
        # Verify the result
        assert result["available"] == True
        assert "available" in result["message"]
    
    @pytest.mark.asyncio
    async def test_check_availability_not_available(self, calendar_tool):
        # Mock the calendar service to return events (not available)
        calendar_tool.service.events().list().execute.return_value = {
            "items": [
                {"id": "event123", "summary": "Existing Meeting"}
            ]
        }
        
        result = await calendar_tool.check_availability("2023-12-31", "14:30")
        
        # Verify the result
        assert result["available"] == False
        assert "not available" in result["message"]
    
    @pytest.mark.asyncio
    async def test_check_availability_no_service(self, calendar_tool):
        # Set service to None to simulate missing service
        calendar_tool.service = None
        
        result = await calendar_tool.check_availability("2023-12-31", "14:30")
        
        # Verify error message is returned
        assert "error" in result
        assert result["error"] == "Calendar service not available."
    
    @pytest.mark.asyncio
    async def test_check_availability_exception(self, calendar_tool):
        # Make the service raise an exception
        calendar_tool.service.events().list().execute.side_effect = Exception("Test error")
        
        result = await calendar_tool.check_availability("2023-12-31", "14:30")
        
        # Verify error message is returned
        assert "error" in result
        assert "Test error" in result["error"] 