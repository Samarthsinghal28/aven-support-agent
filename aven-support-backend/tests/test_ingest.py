import os
import sys
import pytest
import unittest.mock as mock
from bs4 import BeautifulSoup
import hashlib
import asyncio

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ingest

class TestIngest:
    @pytest.fixture
    def mock_env(self):
        with mock.patch.dict(os.environ, {
            "PINECONE_API_KEY": "test_pinecone_key",
            "OPENAI_API_KEY": "test_openai_key",
            "PINECONE_INDEX_NAME": "test-index"
        }):
            yield
    
    def test_fetch_sitemap_urls(self):
        # Mock the requests.get response
        mock_response = mock.MagicMock()
        mock_response.content = """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://www.aven.com/page1</loc>
            </url>
            <url>
                <loc>https://www.aven.com/page2</loc>
            </url>
        </urlset>
        """.encode('utf-8')
        
        with mock.patch('requests.get', return_value=mock_response):
            urls = ingest.fetch_sitemap_urls("https://example.com/sitemap.xml")
            
            # Verify the URLs were extracted correctly
            assert len(urls) == 2
            assert urls[0] == "https://www.aven.com/page1"
            assert urls[1] == "https://www.aven.com/page2"
    
    @pytest.mark.asyncio
    async def test_scrape_page(self):
        # Mock the Playwright page
        mock_page = mock.AsyncMock()
        mock_page.content.return_value = "<html><body>Test content</body></html>"
        
        result = await ingest.scrape_page(mock_page, "https://example.com/page")
        
        # Verify the content was returned
        assert result == "<html><body>Test content</body></html>"
        # Verify the page methods were called
        mock_page.goto.assert_called_once_with("https://example.com/page", wait_until="networkidle", timeout=60000)
        mock_page.wait_for_timeout.assert_called_once_with(2000)
        mock_page.content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scrape_page_exception(self):
        # Mock the Playwright page to raise an exception
        mock_page = mock.AsyncMock()
        mock_page.goto.side_effect = Exception("Test error")
        
        result = await ingest.scrape_page(mock_page, "https://example.com/page")
        
        # Verify empty string is returned on error
        assert result == ""
    
    def test_parse_aven_faqs(self):
        # Create a sample HTML with FAQ structure that matches the actual implementation
        html = """
        <div class="support-list-section">
            <h5>Test Section</h5>
            <ul>
                <li>
                    <a class="title">Test Question?</a>
                    <span>
                        <p>Test Answer</p>
                        <p>More details</p>
                    </span>
                </li>
            </ul>
        </div>
        """
        
        # Call the actual function
        faqs = ingest.parse_aven_faqs(html)
        
        # Verify the FAQs were extracted correctly
        assert len(faqs) == 1
        assert faqs[0]["metadata"]["section"] == "Test Section"
        assert faqs[0]["metadata"]["question"] == "Test Question"
        assert "Test Answer More details" in faqs[0]["text"]
    
    def test_parse_aven_faqs_empty(self):
        # Test with empty HTML
        faqs = ingest.parse_aven_faqs("")
        
        # Verify empty list is returned
        assert faqs == []
    
    def test_fallback_readability(self):
        # Create a sample HTML
        html = "<html><body><article>Test content</article></body></html>"
        
        # Mock the readability Document
        with mock.patch('readability.Document') as mock_document:
            mock_document.return_value.short_title.return_value = "Test Title"
            mock_document.return_value.summary.return_value = "<div>Test content</div>"
            
            # Mock BeautifulSoup
            mock_soup = mock.MagicMock()
            mock_soup.get_text.return_value = "Test Title\n\nTest content"
            
            with mock.patch('bs4.BeautifulSoup', return_value=mock_soup):
                # Mock extract_structured_content
                with mock.patch('ingest.extract_structured_content', return_value=[]):
                    result = ingest.fallback_readability(html, "https://example.com/page")
                    
                    # Verify the content was extracted
                    assert "Test Title" in result
                    assert "Test content" in result
    
    def test_extract_structured_content(self):
        # Create a sample HTML with structured content
        html = """
        <html>
            <body>
                <table>
                    <tr><th>State</th><th>License</th></tr>
                    <tr><td>California</td><td>123456</td></tr>
                </table>
            </body>
        </html>
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Call the actual function
        result = ingest.extract_structured_content(soup)
        
        # Verify the structured content was extracted
        assert len(result) == 1
        assert "License Information" in result[0]
        assert "California" in result[0]
    
    def test_parse_and_chunk(self):
        # Create a sample HTML
        html = """
        <html>
            <body>
                <h1>Test Page</h1>
                <p>This is a test paragraph with some content.</p>
            </body>
        </html>
        """
        
        # Mock the fallback_readability function
        with mock.patch('ingest.fallback_readability', return_value="Test Page\nThis is a test paragraph with some content."):
            # Mock the RecursiveCharacterTextSplitter
            mock_splitter = mock.MagicMock()
            mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
            
            with mock.patch('ingest.RecursiveCharacterTextSplitter', return_value=mock_splitter):
                # Mock datetime.utcnow
                mock_datetime = mock.MagicMock()
                mock_datetime.utcnow.return_value.isoformat.return_value = "2023-01-01T00:00:00"
                
                with mock.patch('ingest.datetime', mock_datetime):
                    chunks = ingest.parse_and_chunk(html, "https://example.com/page")
                    
                    # Verify the chunks were created correctly
                    assert len(chunks) == 2
                    
                    # Check first chunk
                    chunk1_text, chunk1_metadata = chunks[0]
                    assert chunk1_text == "Chunk 1"
                    assert chunk1_metadata["url"] == "https://example.com/page"
                    assert chunk1_metadata["last_crawled"] == "2023-01-01T00:00:00"
                    
                    # Check second chunk
                    chunk2_text, chunk2_metadata = chunks[1]
                    assert chunk2_text == "Chunk 2"
                    assert chunk2_metadata["url"] == "https://example.com/page"
    
    @pytest.mark.asyncio
    async def test_process_chunks_batch(self, mock_env):
        # Create test chunks
        chunks = [
            ("Chunk 1", {"url": "https://example.com/page1"}),
            ("Chunk 2", {"url": "https://example.com/page2"})
        ]
        
        # Mock hashlib.sha256
        mock_hash1 = mock.MagicMock()
        mock_hash1.hexdigest.return_value = "hash1"
        mock_hash2 = mock.MagicMock()
        mock_hash2.hexdigest.return_value = "hash2"
        
        with mock.patch('hashlib.sha256', side_effect=[mock_hash1, mock_hash2]):
            # Mock index.fetch
            mock_fetch_result = mock.MagicMock()
            mock_fetch_result.vectors = {}
            
            with mock.patch.object(ingest.index, 'fetch', return_value=mock_fetch_result):
                # Mock embeddings.embed_documents
                with mock.patch.object(ingest.embeddings, 'embed_documents', return_value=[[0.1] * 1536, [0.2] * 1536]):
                    # Mock index.upsert
                    with mock.patch.object(ingest.index, 'upsert') as mock_upsert:
                        new_chunks, skipped = await ingest.process_chunks_batch(chunks)
                        
                        # Verify the results
                        assert new_chunks == 2
                        assert skipped == 0
                        
                        # Verify upsert was called with correct vectors
                        mock_upsert.assert_called_once()
                        vectors = mock_upsert.call_args[0][0]
                        assert len(vectors) == 2
                        assert vectors[0]["id"] == "hash1"
                        assert vectors[0]["metadata"]["text"] == "Chunk 1"
                        assert vectors[0]["metadata"]["url"] == "https://example.com/page1"
                        assert vectors[0]["values"] == [0.1] * 1536
                        assert vectors[1]["id"] == "hash2"
    
    @pytest.mark.asyncio
    async def test_worker(self, mock_env):
        # Mock the queue
        queue = mock.AsyncMock()
        queue.get.side_effect = [
            "https://example.com/page",  # First item
            asyncio.TimeoutError  # Simulate timeout to end the loop
        ]
        
        # Mock the async_playwright context
        mock_playwright = mock.AsyncMock()
        mock_browser = mock.AsyncMock()
        mock_page = mock.AsyncMock()
        
        mock_playwright.__aenter__.return_value = mock_playwright
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page
        
        with mock.patch('ingest.async_playwright', return_value=mock_playwright):
            # Mock the scrape_page function
            with mock.patch('ingest.scrape_page', return_value="<html><body>Test content</body></html>"):
                # Mock the parse_and_chunk function
                with mock.patch('ingest.parse_and_chunk', return_value=[("Chunk 1", {"url": "https://example.com/page"})]):
                    # Mock the process_chunks_batch function
                    with mock.patch('ingest.process_chunks_batch', return_value=(1, 0)):
                        # Mock asyncio.wait_for to pass through the first call but raise TimeoutError on the second
                        original_wait_for = asyncio.wait_for
                        
                        async def mock_wait_for(coro, timeout):
                            if queue.get.call_count == 1:
                                return await coro
                            raise asyncio.TimeoutError()
                        
                        with mock.patch('asyncio.wait_for', side_effect=mock_wait_for):
                            # Run the worker
                            await ingest.worker(1, queue)
                            
                            # Verify the functions were called correctly
                            queue.get.assert_called()
                            ingest.scrape_page.assert_called_once_with(mock_page, "https://example.com/page")
                            ingest.parse_and_chunk.assert_called_once_with("<html><body>Test content</body></html>", "https://example.com/page")
                            ingest.process_chunks_batch.assert_called_once()
                            queue.task_done.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main(self, mock_env):
        # Mock the fetch_sitemap_urls function
        with mock.patch('ingest.fetch_sitemap_urls', return_value=["https://example.com/page1", "https://example.com/page2"]):
            # Mock the queue
            mock_queue = mock.AsyncMock()
            
            # Mock asyncio.Queue to return our mock queue
            with mock.patch('asyncio.Queue', return_value=mock_queue):
                # Mock asyncio.gather to avoid actually running the workers
                with mock.patch('asyncio.gather') as mock_gather:
                    # Mock async_playwright context
                    mock_playwright = mock.AsyncMock()
                    mock_context = mock.AsyncMock()
                    mock_browser = mock.AsyncMock()
                    mock_page = mock.AsyncMock()
                    
                    mock_playwright.chromium.launch.return_value = mock_browser
                    mock_browser.new_context.return_value = mock_context
                    mock_context.new_page.return_value = mock_page
                    
                    # Mock the async_playwright context manager
                    with mock.patch('ingest.async_playwright') as mock_playwright_func:
                        mock_playwright_func.return_value.__aenter__.return_value = mock_playwright
                        mock_playwright_func.return_value.__aexit__.return_value = None
                        
                        # Mock the queue.join to avoid hanging
                        mock_queue.join = mock.AsyncMock()
                        
                        # Run the main function
                        await ingest.main()
                        
                        # Verify the queue was populated correctly
                        assert mock_queue.put_nowait.call_count == 2
                        mock_queue.put_nowait.assert_any_call("https://example.com/page1")
                        mock_queue.put_nowait.assert_any_call("https://example.com/page2")
                        
                        # Verify the workers were created
                        assert mock_gather.call_count == 1 