import os
import pytest
import unittest.mock as mock
import sys
import os.path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_tools import SerperTool

class TestSerperTool:
    @pytest.fixture
    def serper_tool(self):
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "test_api_key"}):
            tool = SerperTool()
            yield tool
    
    @pytest.mark.asyncio
    async def test_use_success(self, serper_tool):
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
        mock_response.raise_for_status = mock.MagicMock()
        
        with mock.patch('requests.post', return_value=mock_response):
            result = await serper_tool.use("test query")
            
            # Verify the result structure
            assert "organic" in result
            assert len(result["organic"]) == 2
            assert result["organic"][0]["title"] == "First Result"
            assert result["organic"][1]["snippet"] == "This is the second search result snippet."
            assert "answerBox" in result
            assert result["answerBox"]["title"] == "Answer Box"
    
    @pytest.mark.asyncio
    async def test_use_exception(self, serper_tool):
        # Make requests.post raise an exception
        with mock.patch('requests.post', side_effect=Exception("Test error")):
            result = await serper_tool.use("test query")
            
            # Verify error message is returned
            assert "error" in result
            assert "Test error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_use_http_error(self, serper_tool):
        # Mock the requests.post response to raise an HTTP error
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        
        with mock.patch('requests.post', return_value=mock_response):
            result = await serper_tool.use("test query")
            
            # Verify error message is returned
            assert "error" in result
            assert "HTTP Error" in result["error"] 