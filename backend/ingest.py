import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import sys
import hashlib

load_dotenv()

# Configuration
AVEN_SUPPORT_URL = os.getenv('AVEN_SUPPORT_URL', 'https://www.aven.com/support')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aven-support-index')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')  # Faster OpenAI model
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))

# Check for required API keys
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found. Please set it in your .env file.")
    exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
    exit(1)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # For ada-002
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)

# Scrape support page
def scrape_support_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return str(soup)  # Return raw HTML for structured parsing
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ''

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

    print(f"Extracted {len(faqs)} structured FAQs")
    return faqs

# Main ingestion function
def ingest_data(html_content=None):
    if html_content:
        print("Using provided HTML content for ingestion")
        raw_html = html_content
    else:
        print(f"Scraping content from: {AVEN_SUPPORT_URL}")
        raw_html = scrape_support_page(AVEN_SUPPORT_URL)
        if not raw_html:
            print('No content scraped. Please check the URL.')
            return

    # Parse structured FAQs
    structured_chunks = parse_aven_faqs(raw_html)
    if not structured_chunks:
        print('No structured content parsed. Falling back to raw text.')
        raw_text = BeautifulSoup(raw_html, 'html.parser').get_text(strip=True)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(raw_text)
    else:
        # Use structured chunks for splitting if needed
        chunks = [chunk['text'] for chunk in structured_chunks]

    print(f"Processed {len(chunks)} chunks")

    # Embed chunks
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectors = []
    skipped = 0
    
    print("Generating embeddings and checking for duplicates...")
    for i, chunk in enumerate(chunks):
        # Compute content hash
        content_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
        
        # Check if already exists in Pinecone
        existing = index.query(
            vector=[0] * 1536,  # Dummy vector
            filter={
                'content_hash': content_hash
            },
            top_k=1
        )
        
        if existing['matches']:
            print(f"  Skipping duplicate chunk {i} (hash: {content_hash[:10]}...)")
            skipped += 1
            continue
        
        embedding = embeddings.embed_query(chunk)
        metadata = {
            'text': chunk,
            'source': AVEN_SUPPORT_URL,
            'content_hash': content_hash
        }
        if i < len(structured_chunks):
            metadata.update(structured_chunks[i]['metadata'])
            
        vectors.append({
            'id': f'chunk_{content_hash}',  # Use hash in ID for uniqueness
            'values': embedding,
            'metadata': metadata
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(chunks)} chunks (skipped {skipped})")

    if not vectors:
        print("No new content to ingest - all chunks already exist.")
        return

    # Upsert to Pinecone
    print("Uploading to Pinecone...")
    index.upsert(vectors)
    print(f'Successfully ingested {len(vectors)} new chunks (skipped {skipped} duplicates) into Pinecone index: {INDEX_NAME}')

if __name__ == '__main__':
    html_content = None
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
        with open(html_file, 'r') as f:
            html_content = f.read()
    ingest_data(html_content) 