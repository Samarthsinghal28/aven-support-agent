 
import asyncio
import hashlib
import requests
from xml.etree import ElementTree
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from readability import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import os

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
SITEMAP_URL     = "https://www.aven.com/sitemap.xml"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "aven-support-index")
EMBED_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", 500))
CONCURRENCY      = int(os.getenv("CONCURRENCY", 5))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", 50))

# Check for required API keys
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found. Please set it in your .env file.")
    exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
    exit(1)

# ─── Pinecone Init ─────────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

# ─── Helpers ─────────────────────────────────────────────────────────────────
def fetch_sitemap_urls(url=SITEMAP_URL):
    """Fetch all URLs from the sitemap"""
    print(f"Fetching sitemap from: {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    tree = ElementTree.fromstring(r.content)
    urls = [loc.text for loc in tree.findall(".//{*}loc")]
    print(f"Found {len(urls)} URLs in sitemap")
    return urls

async def scrape_page(page, url):
    """Scrape a single page using an existing Playwright page"""
    try:
        await page.goto(url, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(2000)
        return await page.content()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def parse_aven_faqs(html_content):
    """Extract structured FAQ data from Aven support page"""
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    faqs = []

    # Find all support sections
    sections = soup.find_all('div', class_='support-list-section')
    
    for section in sections:
        # Get section title (e.g., 'Trending Articles', 'Payments')
        section_title = section.find('h5')
        if section_title:
            section_name = section_title.get_text(strip=True)
        else:
            section_name = 'Uncategorized'

        # Find all FAQ items in the section
        items = section.find_all('li')
        
        for item in items:
            # Get question
            question_elem = item.find('a', class_='title')
            if question_elem:
                question = question_elem.get_text(strip=True).replace('?', '').strip()
            else:
                continue

            # Get answer (inside <span>)
            answer_span = item.find('span')
            if answer_span:
                # Extract all text from paragraphs, lists, etc.
                answer_parts = []
                for elem in answer_span.find_all(['p', 'ul', 'ol']):
                    text = elem.get_text(strip=True)
                    if text:
                        answer_parts.append(text)
                answer = ' '.join(answer_parts)
            else:
                answer = ''

            if question and answer:
                # Create structured chunk
                chunk = f"Section: {section_name}\nQuestion: {question}\nAnswer: {answer}"
                faqs.append({
                    'text': chunk,
                    'metadata': {
                        'section': section_name,
                        'question': question
                    }
                })

    return faqs

def fallback_readability(html, url):
    """Extract main content using readability with improved text formatting for tables and lists"""
    doc = Document(html, url=url)
    title = doc.short_title()
    content = doc.summary()  # HTML of main content
    
    # Parse with BeautifulSoup to preserve structure
    soup = BeautifulSoup(content, "html.parser")
    
    # First, extract any structured content (license tables, etc.)
    structured_sections = extract_structured_content(soup)
    
    # Handle remaining tables with generic formatting
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
            if cells:  # Only add non-empty rows
                rows.append(" | ".join(cells))
        
        if rows:
            table_text = "\n".join(rows)
            # Replace the table with formatted text
            table.replace_with(soup.new_string(f"\n\nTable:\n{table_text}\n\n"))
    
    # Handle lists with proper formatting
    for ul in soup.find_all("ul"):
        items = []
        for li in ul.find_all("li"):
            item_text = li.get_text(strip=True)
            if item_text:
                items.append(f"• {item_text}")
        
        if items:
            list_text = "\n".join(items)
            ul.replace_with(soup.new_string(f"\n{list_text}\n"))
    
    for ol in soup.find_all("ol"):
        items = []
        for i, li in enumerate(ol.find_all("li"), 1):
            item_text = li.get_text(strip=True)
            if item_text:
                items.append(f"{i}. {item_text}")
        
        if items:
            list_text = "\n".join(items)
            ol.replace_with(soup.new_string(f"\n{list_text}\n"))
    
    # Add proper spacing around headers
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        header_text = header.get_text(strip=True)
        if header_text:
            header.replace_with(soup.new_string(f"\n\n{header_text}\n"))
    
    # Handle divs and sections with proper spacing
    for div in soup.find_all(["div", "section", "article"]):
        # Add spacing between major content blocks
        if div.get_text(strip=True):
            div.append(soup.new_string("\n"))
    
    # Clean up paragraphs
    for p in soup.find_all("p"):
        p_text = p.get_text(strip=True)
        if p_text:
            p.replace_with(soup.new_string(f"{p_text}\n\n"))
    
    text = soup.get_text()
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split('\n')]
    # Remove empty lines but preserve intentional spacing
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned_lines.append("")
            prev_empty = True
    
    clean_text = '\n'.join(cleaned_lines).strip()
    
    # Combine structured sections with the main text
    all_content = []
    if title:
        all_content.append(title)
    
    if structured_sections:
        all_content.extend(structured_sections)
    
    if clean_text:
        all_content.append(clean_text)
    
    return '\n\n'.join(all_content)

def extract_structured_content(soup):
    """Extract and format structured content like license tables, regulations, etc."""
    structured_sections = []
    
    # Look for license/regulation tables specifically
    for table in soup.find_all("table"):
        # Check if this looks like a license table
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if any(keyword in ' '.join(headers) for keyword in ['state', 'license', 'registration', 'permit']):
            
            rows = []
            for row in table.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                if cells and any(cell.strip() for cell in cells):  # Non-empty row
                    rows.append(cells)
            
            if rows:
                # Format as a proper license table
                formatted_table = "License Information:\n"
                
                # If we have headers, use them
                if headers and len(headers) >= 2:
                    for row in rows[1:] if headers else rows:  # Skip header row if present
                        if len(row) >= 2:
                            state = row[0].strip()
                            license_info = ' | '.join(cell.strip() for cell in row[1:] if cell.strip())
                            if state and license_info:
                                formatted_table += f"{state}: {license_info}\n"
                else:
                    # No clear headers, try to infer structure
                    for row in rows:
                        if len(row) >= 2:
                            key = row[0].strip()
                            values = ' | '.join(cell.strip() for cell in row[1:] if cell.strip())
                            if key and values:
                                formatted_table += f"{key}: {values}\n"
                
                structured_sections.append(formatted_table)
                # Remove the table so it doesn't get processed again
                table.decompose()
    
    # Look for definition lists (dt/dd pairs)
    for dl in soup.find_all("dl"):
        items = []
        current_term = None
        
        for child in dl.children:
            if child.name == "dt":
                current_term = child.get_text(strip=True)
            elif child.name == "dd" and current_term:
                definition = child.get_text(strip=True)
                if definition:
                    items.append(f"{current_term}: {definition}")
                current_term = None
        
        if items:
            structured_sections.append("\n".join(items))
            dl.decompose()
    
    return structured_sections

def parse_and_chunk(html, url):
    """Parse HTML and return chunks with metadata"""
    # 1) Attempt structured FAQ parser for support pages
    if "/support" in url:
        faqs = parse_aven_faqs(html)
        if faqs:
            chunks = []
            for faq in faqs:
                metadata = {
                    "url": url,
                    "last_crawled": datetime.utcnow().isoformat(),
                    **faq["metadata"]
                }
                chunks.append((faq["text"], metadata))
            return chunks
    
    # 2) Fallback to Readability for all other pages
    text = fallback_readability(html, url)
    if len(text.strip()) < 100:  # Skip pages with minimal content
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=50, 
        length_function=len
    )
    chunks = splitter.split_text(text)
    
    return [(chunk, {
        "url": url,
        "last_crawled": datetime.utcnow().isoformat()
    }) for chunk in chunks if chunk.strip()]

async def process_chunks_batch(chunks_batch):
    """Process a batch of chunks: check duplicates, embed, and upsert"""
    if not chunks_batch:
        return 0, 0
    
    # Check for existing chunks
    chunk_hashes = [hashlib.sha256(text.encode()).hexdigest() for text, _ in chunks_batch]
    existing = index.fetch(ids=chunk_hashes)
    existing_ids = set(existing.vectors.keys()) if existing.vectors else set()
    
    # Filter out duplicates
    new_chunks = [(text, meta, h) for (text, meta), h in zip(chunks_batch, chunk_hashes) 
                  if h not in existing_ids]
    
    if not new_chunks:
        return 0, len(chunks_batch)
    
    # Batch embed all new chunks
    texts_to_embed = [text for text, _, _ in new_chunks]
    embeddings_batch = embeddings.embed_documents(texts_to_embed)
    
    # Create vectors for upsert
    vectors = []
    for (text, metadata, chunk_hash), embedding in zip(new_chunks, embeddings_batch):
        vectors.append({
            "id": chunk_hash,
            "values": embedding,
            "metadata": {
                **metadata,
                "text": text,
                "content_hash": chunk_hash
            }
        })
    
    # Upsert batch
    if vectors:
        index.upsert(vectors)
    
    return len(vectors), len(chunks_batch) - len(vectors)

async def worker(worker_id, queue):
    """Worker that processes URLs from the queue"""
    processed = 0
    total_new_chunks = 0
    total_skipped = 0
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            chunks_batch = []
            
            while True:
                try:
                    # Get URL from queue with timeout to avoid hanging
                    url = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                
                try:
                    print(f"Worker {worker_id}: Processing {url}")
                    html = await scrape_page(page, url)
                    
                    if html:
                        chunks = parse_and_chunk(html, url)
                        chunks_batch.extend(chunks)
                        
                        # Process batch when it reaches BATCH_SIZE
                        if len(chunks_batch) >= BATCH_SIZE:
                            new_count, skipped_count = await process_chunks_batch(chunks_batch)
                            total_new_chunks += new_count
                            total_skipped += skipped_count
                            chunks_batch = []
                    
                    processed += 1
                    if processed % 10 == 0:
                        print(f"Worker {worker_id}: Processed {processed} pages")
                    
                except Exception as e:
                    print(f"Worker {worker_id}: Error processing {url}: {e}")
                finally:
                    queue.task_done()
            
            # Process remaining chunks in batch
            if chunks_batch:
                new_count, skipped_count = await process_chunks_batch(chunks_batch)
                total_new_chunks += new_count
                total_skipped += skipped_count
                
        finally:
            await browser.close()
    
    print(f"Worker {worker_id}: Completed {processed} pages, {total_new_chunks} new chunks, {total_skipped} skipped")
    return total_new_chunks, total_skipped

# ─── Main Entrypoint ──────────────────────────────────────────────────────────
async def main():
    """Main function to orchestrate the ingestion pipeline"""
    print("Starting Aven content ingestion pipeline...")
    
    try:
        urls = fetch_sitemap_urls()
        
        # Create queue and add all URLs
        queue = asyncio.Queue()
        for url in urls:
            queue.put_nowait(url)
        
        print(f"Starting {CONCURRENCY} workers...")
        
        # Create and start workers
        workers = [
            asyncio.create_task(worker(i, queue))
            for i in range(CONCURRENCY)
        ]
        
        # Wait for all work to complete
        await queue.join()
        
        # Collect results from workers
        results = await asyncio.gather(*workers, return_exceptions=True)
        
        # Calculate totals
        total_new = sum(r[0] for r in results if isinstance(r, tuple))
        total_skipped = sum(r[1] for r in results if isinstance(r, tuple))
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(f"Total URLs processed: {len(urls)}")
        print(f"New chunks ingested: {total_new}")
        print(f"Duplicate chunks skipped: {total_skipped}")
        print(f"Pinecone index: {INDEX_NAME}")
        print("="*60)
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())