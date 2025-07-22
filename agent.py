import os
import requests
import logging
import asyncio
from dotenv import load_dotenv
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from crewai import Agent, Task, Crew, Process
from langchain_openai import OpenAIEmbeddings, OpenAIModerationChain
from pinecone import Pinecone
from serpapi import search

load_dotenv()

# Configuration and LLM Selection
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_API_KEY')
PINECONE_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aven-support-index')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
SEARCH_SITE = os.getenv('SEARCH_SITE', 'aven.com')
TOP_K = int(os.getenv('TOP_K', 5))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required API keys
if not OPENAI_KEY and not GEMINI_KEY:
    logger.warning("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env file.")

if OPENAI_KEY:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_KEY)
    moderation = None  # Simplified for now
elif GEMINI_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=GEMINI_KEY)
    moderation = None
else:
    llm = None
    moderation = None

# Initialize Pinecone only if API key is available
if PINECONE_KEY:
    pc = Pinecone(api_key=PINECONE_KEY)
    index = pc.Index(INDEX_NAME)
    if OPENAI_KEY:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_KEY)
    else:
        embeddings = None
else:
    logger.warning("No Pinecone API key found. Set PINECONE_API_KEY in .env file.")
    pc = None
    index = None
    embeddings = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@lru_cache(maxsize=100)
def retrieve_from_pinecone(query):
    if not index or not embeddings:
        logger.error("Knowledge base not available - missing API keys")
        return ["Knowledge base not available - missing API keys"]
    query_emb = embeddings.embed_query(query)
    results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
    logger.info(f"Retrieved {len(results['matches'])} results from Pinecone")
    return [match['metadata']['text'] for match in results['matches']]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fallback_search(query):
    if not SERPAPI_KEY:
        logger.error("Web search not available - missing SERPAPI_API_KEY")
        return ["Web search not available - missing SERPAPI_API_KEY"]
    params = {'q': query + f' site:{SEARCH_SITE}', 'api_key': SERPAPI_KEY}
    results = search(params)
    organic_results = results.get('organic_results', [])
    if not organic_results:
        return ['No results found. Suggest scheduling a meeting.']
    logger.info(f"Fallback search returned {len(organic_results)} results")
    return [res['snippet'] for res in organic_results[:TOP_K]]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def async_retrieve_from_pinecone(query):
    return retrieve_from_pinecone(query)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def async_fallback_search(query):
    return fallback_search(query)

# Only create agents if LLM is available
if llm:
    # Add Guardrails Agent with enhanced moderation
    guardrails_agent = Agent(
        role='Guardrails Checker',
        goal='Detect sensitive topics like personal data, legal advice, or toxicity',
        backstory='You ensure safe and appropriate responses using moderation tools.',
        llm=llm,
        verbose=True
    )

    # Agents
    router_agent = Agent(
        role='Query Router',
        goal='Determine if query can be answered from knowledge base or needs fallback',
        backstory='You classify user queries and route them appropriately.',
        llm=llm,
        verbose=True
    )

    retrieval_agent = Agent(
        role='Knowledge Retriever',
        goal='Retrieve relevant information from Pinecone',
        backstory='You fetch accurate info from the Aven support knowledge base.',
        llm=llm,
        verbose=True
    )

    search_agent = Agent(
        role='Web Searcher',
        goal='Perform fallback search if RAG fails',
        backstory='You search the web for additional information on Aven.',
        llm=llm,
        verbose=True
    )

    response_agent = Agent(
        role='Response Generator',
        goal='Generate friendly, accurate responses with citations',
        backstory='You compile information into helpful responses.',
        llm=llm,
        verbose=True
    )

    # Tasks
    def create_tasks(query):
        guard_task = Task(
            description=f'Check if this query is sensitive (personal data, legal advice, toxicity, etc.): {query}. If sensitive, respond with "I cannot assist with that." else pass to next.',
            agent=guardrails_agent
        )
        route_task = Task(
            description=f'Analyze the query: {query} and decide routing.',
            agent=router_agent
        )
        retrieve_task = Task(
            description=f'Retrieve info for: {query}',
            agent=retrieval_agent,
            tools=[retrieve_from_pinecone]
        )
        search_task = Task(
            description=f'Search web for: {query}',
            agent=search_agent,
            tools=[fallback_search]
        )
        response_task = Task(
            description=f'Generate response for: {query} using retrieved info.',
            agent=response_agent
        )
        return [guard_task, route_task, retrieve_task, search_task, response_task]

    # Main function
    def run_agent(query):
        if not query or not isinstance(query, str) or len(query) < 3:
            logger.warning("Invalid query provided")
            return "Invalid query. Please provide a detailed question."
        tasks = create_tasks(query)
        crew = Crew(agents=[router_agent, retrieval_agent, search_agent, response_agent], tasks=tasks, verbose=2, process=Process.parallel)
        result = crew.kickoff()
        return result
else:
    def create_tasks(query):
        return []
    
    def run_agent(query):
        return "Agent system not available - missing API keys. Please check your .env file."

if __name__ == '__main__':
    if not llm:
        print("Cannot run agent - missing required API keys. Please set up your .env file.")
    else:
        query = input('Enter your question: ')
        response = run_agent(query)
        print(response)