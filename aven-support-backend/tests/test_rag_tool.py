import os
import pytest
import unittest.mock as mock
import sys
import os.path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_tools import RAGTool

class TestRAGTool:
    @pytest.fixture
    def rag_tool(self):
        with mock.patch('pinecone.Pinecone') as mock_pinecone:
            mock_index = mock.MagicMock()
            mock_pinecone.return_value.Index.return_value = mock_index
            mock_pinecone.return_value.list_indexes.return_value.names.return_value = ['test-index']
            
            with mock.patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
                mock_embeddings.return_value.embed_query.return_value = [0.1] * 1536
                
                tool = RAGTool()
                tool.index = mock_index
                tool.embeddings = mock_embeddings.return_value
                yield tool
    
    @pytest.mark.asyncio
    async def test_use_success(self, rag_tool):
        # Mock the index query response
        rag_tool.index.query.return_value = {
            'matches': [
                {
                    'metadata': {
                        'text': 'This is the first relevant text.',
                        'url': 'https://example.com/1'
                    },
                    'score': 0.95
                },
                {
                    'metadata': {
                        'text': 'This is the second relevant text.',
                        'url': 'https://example.com/2'
                    },
                    'score': 0.85
                }
            ]
        }
        
        result = await rag_tool.use("test query")
        
        # Verify the result contains the expected contexts
        assert 'contexts' in result
        assert len(result['contexts']) == 2
        assert result['contexts'][0] == 'This is the first relevant text.'
        assert result['contexts'][1] == 'This is the second relevant text.'
        
        # Verify the embeddings and query were called correctly
        rag_tool.embeddings.embed_query.assert_called_once_with("test query")
        rag_tool.index.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_use_no_index(self, rag_tool):
        # Set index to None to simulate missing index
        rag_tool.index = None
        
        result = await rag_tool.use("test query")
        
        # Verify error message is returned
        assert 'error' in result
        assert result['error'] == 'Pinecone index not found.'
    
    @pytest.mark.asyncio
    async def test_use_exception(self, rag_tool):
        # Make the index query raise an exception
        rag_tool.index.query.side_effect = Exception("Test error")
        
        result = await rag_tool.use("test query")
        
        # Verify error message is returned
        assert 'error' in result
        assert "Test error" in result['error'] 