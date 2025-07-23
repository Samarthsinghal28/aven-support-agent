import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aven-support-index')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
TOP_K = int(os.getenv('TOP_K', 5))

# Check for required API keys
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found. Please set it in your .env file.")
    exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
    exit(1)

# Initialize Pinecone and Embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
try:
    index = pc.Index(INDEX_NAME)
    # Test if index is accessible
    index.describe_index_stats()
    print(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Error connecting to Pinecone index '{INDEX_NAME}': {e}")
    print("Please run ingest.py first to create and populate the index.")
    exit(1)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def query_rag(question):
    print(f"Searching for: {question}")
    
    # Embed question
    query_embedding = embeddings.embed_query(question)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    
    # Extract relevant texts
    contexts = []
    for match in results['matches']:
        contexts.append({
            'text': match['metadata']['text'],
            'score': match['score']
        })
    
    return contexts

if __name__ == '__main__':
    print("RAG Query System for Aven Support")
    print("=" * 40)
    
    while True:
        question = input('\nEnter your question about Aven (or "quit" to exit): ')
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        results = query_rag(question)
        
        if not results:
            print("No relevant information found.")
            continue
            
        print(f"\nFound {len(results)} relevant contexts:")
        print("-" * 40)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:200]}...")
            print() 