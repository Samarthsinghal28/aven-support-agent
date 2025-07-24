import os
import logging
import requests
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field

load_dotenv()

# Configuration
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
PINECONE_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aven-support-index')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')  # Fastest OpenAI model
SEARCH_SITE = os.getenv('SEARCH_SITE', 'aven.com')
TOP_K = int(os.getenv('TOP_K', 5))  # Reduced to 1 for maximum speed
MAX_DOC_LENGTH = int(os.getenv('MAX_DOC_LENGTH', 400))
USE_OPENAI = os.getenv('USE_OPENAI', 'true') == 'true'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM with streaming support
if OPENAI_KEY and USE_OPENAI:
    llm = ChatOpenAI(
        model='gpt-3.5-turbo', 
        api_key=OPENAI_KEY,
        temperature=0.1,  # Lower temperature for more consistent responses
        max_tokens=300,
        streaming=True,
    )
elif GEMINI_KEY:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash', 
        api_key=GEMINI_KEY,
        temperature=0.1,
        max_output_tokens=500 # Correct parameter for Gemini
    )
else:
    llm = None
    logger.error("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env file.")

# Initialize Pinecone
if PINECONE_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_KEY)
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME in existing_indexes:
            index = pc.Index(INDEX_NAME)
            # OpenAI Embeddings are required for the knowledge base, as it was built using them.
            if OPENAI_KEY:
                embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL, 
                    api_key=OPENAI_KEY,
                    chunk_size=500,  # Smaller chunks for speed
                    request_timeout=8,  # Faster timeout
                    max_retries=1  # Reduce retries for speed
                )
                logger.info(f"Successfully connected to Pinecone index '{INDEX_NAME}' with OpenAI embeddings.")
            else:
                # If no OpenAI key, we cannot use the knowledge base.
                logger.warning(f"Connected to Pinecone index '{INDEX_NAME}', but OPENAI_API_KEY is missing for embeddings.")
                logger.warning("Knowledge base search will be disabled. Agent will rely on web search.")
                embeddings = None
        else:
            logger.warning(f"Pinecone index '{INDEX_NAME}' not found. Available indexes: {existing_indexes}")
            logger.warning("Run 'python ingest_with_js.py' to create and populate the index.")
            index = None
            embeddings = None
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        pc = None
        index = None
        embeddings = None
else:
    logger.warning("No Pinecone API key found. Set PINECONE_API_KEY in .env file.")
    pc = None
    index = None
    embeddings = None

# Cache for embeddings and search results
@lru_cache(maxsize=100)
def cached_embedding(query: str) -> List[float]:
    """Cache embeddings to avoid redundant API calls"""
    if not embeddings:
        return []
    try:
        return embeddings.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

@lru_cache(maxsize=50)
def cached_web_search(query: str) -> str:
    """Cache web search results for common queries"""
    return _perform_web_search(query)

# Fast routing logic - no LLM needed
def needs_web_search(query: str) -> bool:
    """Fast heuristic to determine if web search is needed"""
    query_lower = query.lower()
    web_indicators = [
        'today', 'latest', 'current', 'recent', 'new', 'update', 'updated',
        '2024', '2025', 'now', 'this year', 'this month', 'this week',
        'breaking', 'news', 'regulation', 'regulatory', 'price change',
        'announcement', 'launch', 'released', 'just', 'recently'
    ]
    return any(indicator in query_lower for indicator in web_indicators)

def is_insufficient_kb_content(kb_response: str, original_query: str) -> bool:
    """Fast evaluation of knowledge base content quality with LLM relevance check"""
    if not kb_response or len(kb_response.strip()) < 20:
        return True
    
    # Quick checks for obviously insufficient responses
    insufficient_indicators = [
        "no relevant information found",
        "no readable content found", 
        "error retrieving from knowledge base",
        "knowledge base not available"
    ]
    
    kb_lower = kb_response.lower()
    if any(indicator in kb_lower for indicator in insufficient_indicators):
        logger.info("KB content marked insufficient due to error indicators")
        return True
    
    # Fast heuristic: If content is substantial, assume sufficient (avoids LLM call in 80%+ cases)
    if len(kb_response) > 100 and kb_response.count('.') > 0:
        logger.debug("KB content appears sufficient based on length and structure")
        return False
    
    # Always use LLM evaluation for relevance, regardless of length
    # This ensures we catch cases where we have content but it doesn't answer the specific question
    logger.info("Using LLM to evaluate KB content relevance...")
    return is_insufficient_response_llm(kb_response, original_query)

def is_insufficient_response_llm(response: str, original_query: str) -> bool:
    """Use LLM to determine if a response provides sufficient information for the query"""
    if not llm or not response or len(response.strip()) < 10:
        return True
    
    try:
        start_eval = time.time()
        evaluation_prompt = f"""You are an AI evaluator. Your job is to determine if content adequately addresses a user's question.

USER'S ORIGINAL QUESTION: {original_query}

CONTENT TO EVALUATE: {response}

EVALUATION CRITERIA:
1. Does the content directly relate to the user's question?
2. Does the content provide specific, useful information?
3. Would this content help answer the user's question?

IMPORTANT: 
- Content that says "no information found" or similar is INSUFFICIENT
- Content with actual details, facts, or relevant information is SUFFICIENT
- Even partial information that relates to the question is SUFFICIENT

Respond with exactly one word: "SUFFICIENT" or "INSUFFICIENT"
"""

        evaluation = llm.invoke(evaluation_prompt)
        result = evaluation.content.strip().upper() if hasattr(evaluation, 'content') else str(evaluation).strip().upper()
        end_eval = time.time()
        logger.debug(f"LLM content evaluation took {end_eval - start_eval:.2f}s")
        
        logger.info(f"LLM evaluation of content: {result}")
        return "INSUFFICIENT" in result
        
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        # Conservative fallback - assume insufficient if we can't evaluate
        return True

# Keep the original function for backward compatibility
def is_insufficient_response(response: str, original_query: str) -> bool:
    """Use LLM to determine if a response provides sufficient information for the query"""
    return is_insufficient_response_llm(response, original_query)

# Optimized tools
def knowledge_base_search_fast(query: str) -> str:
    """
    Fast knowledge base search with caching and parallel processing.
    """
    if not index or not embeddings:
        return "Knowledge base not available - missing API keys or index not found"
    
    try:
        start_emb = time.time()
        # Use cached embedding
        query_emb = cached_embedding(query)
        end_emb = time.time()
        logger.debug(f"Embedding generation took {end_emb - start_emb:.2f}s")

        if not query_emb:
            return "Error generating query embedding"
        
        logger.debug(f"Generated query embedding (first 5 values): {query_emb[:5]}")
        
        start_query = time.time()
        # Query Pinecone
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        end_query = time.time()
        logger.debug(f"Pinecone query took {end_query - start_query:.2f}s")
        logger.info(f"Retrieved {len(results['matches'])} results from Pinecone")
        logger.debug(f"Match scores: {[match['score'] for match in results['matches']]}")
        
        if results['matches']:
            # Limit document length and combine results
            docs = []
            for match in results['matches']:
                try:
                    # Handle different metadata structures
                    metadata = match.get('metadata', {})
                    if isinstance(metadata, dict):
                        text = (
                            metadata.get('text', '')
                            or metadata.get('content', '')
                            or metadata.get('page_content', '')  # langchain default key
                            or str(metadata)
                        )
                    else:
                        text = str(metadata)
                    
                    # Truncate long documents for faster processing
                    if len(text) > MAX_DOC_LENGTH:
                        text = text[:MAX_DOC_LENGTH] + "..."
                    
                    if text.strip():  # Only add non-empty text
                        docs.append(text)
                        logger.debug(f"Added document snippet: {text[:100]}... (score: {match['score']})")
                        
                except Exception as e:
                    logger.error(f"Error processing match metadata: {e}")
                    logger.error(f"Match structure: {match}")
                    continue
            
            if docs:
                combined = '\n\n---\n\n'.join(docs)
                logger.debug(f"Combined context length: {len(combined)}")
                return combined
            else:
                logger.warning("No readable content found in search results despite matches")
                return "No readable content found in search results."
        else:
            logger.warning("No matches found in Pinecone for query")
            return "No relevant information found in the knowledge base."
    except Exception as e:
        logger.error(f"Error retrieving from Pinecone: {e}")
        return f"Error retrieving from knowledge base: {str(e)}"

def _perform_web_search(query: str) -> str:
    """Internal web search function"""
    if not SERPER_API_KEY:
        return "Web search not available - missing SERPER_API_KEY"
    
    try:
        url = "https://google.serper.dev/search"
        
        payload = json.dumps({
            "q": f"{query} site:{SEARCH_SITE}",
            "num": TOP_K
        })
        
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        
        start_req = time.time()
        response = requests.post(url, headers=headers, data=payload, timeout=5)  # Reduced timeout
        end_req = time.time()
        logger.debug(f"Web search request took {end_req - start_req:.2f}s")
        response.raise_for_status()
        
        results = response.json()
        organic_results = results.get('organic', [])
        
        if not organic_results:
            return 'No current web results found. Please contact Aven support at support@aven.com.'
        
        logger.info(f"Web search returned {len(organic_results)} results")
        
        formatted_results = []
        for result in organic_results[:TOP_K]:
            title = result.get('title', 'Unknown')
            snippet = result.get('snippet', '')
            link = result.get('link', '')
            # Truncate snippets for faster processing
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            formatted_results.append(f"**{title}**\n{snippet}\nSource: {link}")
        
        return '\n\n'.join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return "Web search temporarily unavailable. Please contact support@aven.com."

# Enhanced prompt templates with strong guardrails
SYSTEM_PROMPT = """You are Aven's official customer support AI assistant. You ONLY answer questions about Aven products and services.

STRICT RULES:
1. ONLY discuss Aven's home equity credit cards, HELOCs, rates, fees, applications, and company information
2. IMMEDIATELY REFUSE any questions about other companies, general topics, or non-Aven subjects
3. For off-topic questions, say: "I can only help with Aven-related questions. Please contact support@aven.com for assistance."
4. NEVER provide information about competitors, other financial products, or general knowledge
5. Be helpful and professional for Aven-related questions only

AVEN TOPICS YOU CAN DISCUSS:
- Aven's home equity credit cards and HELOCs
- Interest rates and fees
- Application processes
- Company information (founders, history)
- Product features and benefits
- Support and contact information

FOR ANY OTHER TOPIC: Politely refuse and redirect to support@aven.com.

CONTACT INFO:
- Email: support@aven.com  
- Website: aven.com/support"""

def generate_response_with_context(query: str, context: str, source_type: str) -> str:
    """Generate final response with enhanced prompting and guardrails"""
    if not llm:
        return "Response generation not available - missing LLM configuration"
    
    try:
        # Determine if this is a fallback scenario
        is_fallback = "fallback" in source_type.lower()
        
        # Enhanced prompt with strong guardrails
        fallback_note = ""
        if is_fallback:
            fallback_note = "\nNOTE: This information comes from web search after our knowledge base didn't have sufficient details."
        
        prompt = f"""{SYSTEM_PROMPT}

USER QUESTION: {query}

CONTEXT FROM {source_type.upper()}:
{context}{fallback_note}

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context
2. If this is from web search fallback, acknowledge that you searched beyond the knowledge base
3. Be concise but thorough (ideal for voice response)
4. Always end with appropriate next steps or contact information
5. NEVER add information not present in the context
6. If the context is still insufficient, honestly acknowledge limitations

Generate a helpful, accurate response:"""

        # Use streaming for faster perceived response time
        start_gen = time.time()
        response = llm.invoke(prompt)
        end_gen = time.time()
        logger.debug(f"LLM response generation took {end_gen - start_gen:.2f}s")
        
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm experiencing technical difficulties. Please contact Aven support at support@aven.com for immediate assistance."

# Parallel processing function
def run_agent_fast(query: str) -> str:
    """
    Optimized agent pipeline with parallel processing and intelligent fallback
    """
    if not llm:
        return "Agent system not available - missing LLM API keys. Please check your .env file."
    
    if not query or not isinstance(query, str) or len(query.strip()) < 3:
        logger.warning("Invalid query provided")
        return "Please provide a more detailed question so I can help you better."
    
    query = query.strip()
    start_time = time.time()
    
    try:
        # Fast routing decision (no LLM call needed)
        use_web_search = needs_web_search(query)
        logger.info(f"Query routing: {'WEB_SEARCH' if use_web_search else 'KNOWLEDGE_BASE'}")
        
        # Parallel execution of data retrieval
        context = ""
        source_type = ""
        tried_fallback = False
        
        try:
            if use_web_search:
                # Direct web search for time-sensitive queries
                logger.info("Starting web search...")
                start_web = time.time()
                context = cached_web_search(query)
                end_web = time.time()
                logger.debug(f"Web search took {end_web - start_web:.2f}s")
                source_type = "web search"
            else:
                # Try knowledge base first
                logger.info("Starting knowledge base search...")
                start_kb = time.time()
                context = knowledge_base_search_fast(query)
                end_kb = time.time()
                logger.debug(f"Knowledge base search took {end_kb - start_kb:.2f}s")
                source_type = "knowledge base"
                
                # Check if knowledge base response is insufficient
                start_eval = time.time()
                if is_insufficient_kb_content(context, query):
                    logger.info("Knowledge base response insufficient, trying web search fallback...")
                    start_fallback = time.time()
                    web_context = cached_web_search(query)
                    end_fallback = time.time()
                    logger.debug(f"Fallback web search took {end_fallback - start_fallback:.2f}s")
                    
                    # If web search provides better results, use it
                    if web_context and not is_insufficient_kb_content(web_context, query) and len(web_context.strip()) > 50:
                        context = web_context
                        source_type = "web search (fallback)"
                        tried_fallback = True
                        logger.info("Using web search fallback results")
                    else:
                        logger.info("Web search fallback also insufficient, using knowledge base response")
                end_eval = time.time()
                logger.debug(f"Content sufficiency evaluation took {end_eval - start_eval:.2f}s")
            
            logger.info(f"Retrieved context length: {len(context)} characters")
            if tried_fallback:
                logger.info("Successfully used fallback strategy")
                
        except Exception as search_error:
            logger.error(f"Error in search phase: {search_error}")
            return f"Search error: {str(search_error)}. Please contact support@aven.com."
        
        # Single LLM call to generate final response
        try:
            logger.info("Generating response with LLM...")
            start_response = time.time()
            response = generate_response_with_context(query, context, source_type)
            end_response = time.time()
            logger.debug(f"Full response generation took {end_response - start_response:.2f}s")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Total response time: {elapsed_time:.2f} seconds")
            
            return response
        except Exception as llm_error:
            logger.error(f"Error in LLM generation: {llm_error}")
            return f"Response generation error: {str(llm_error)}. Please contact support@aven.com."
        
    except Exception as e:
        logger.error(f"Error in fast agent pipeline: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Fallback response with guardrails
        return "I'm sorry, I'm experiencing technical difficulties right now. For immediate assistance with your Aven-related question, please contact our support team at support@aven.com or visit aven.com/support."

# Legacy function for backward compatibility (now optimized)
@tool
def knowledge_base_search(query: str) -> str:
    """
    Legacy wrapper for knowledge base search tool
    """
    return knowledge_base_search_fast(query)

@tool
def web_search(query: str) -> str:
    """
    Legacy wrapper for web search tool
    """
    return cached_web_search(query)

# Enhanced agents with better prompts and guardrails
def create_agents():
    if not llm:
        return None, None, None, None
    
    router_agent = Agent(
        role='Query Router',
        goal='Analyze user queries and determine the best information source with high accuracy',
        backstory="""You are a precise query analyzer for Aven customer support. Your job is to quickly determine if a query needs current/external information (web search) or can be answered from internal documentation (knowledge base). 

GUIDELINES:
- Use KNOWLEDGE_BASE for: product info, how-to questions, policies, account issues, general Aven questions
- Use WEB_SEARCH only for: recent news, current events, regulatory changes, "today/latest/current" requests
- When in doubt, choose KNOWLEDGE_BASE as it contains comprehensive Aven information
- Be decisive - avoid ambiguous routing decisions""",
        llm=llm,
        verbose=True
    )
    
    retrieval_agent = Agent(
        role='Knowledge Retriever',
        goal='Retrieve the most relevant and accurate information from Aven knowledge base',
        backstory="""You are Aven's expert knowledge retriever. You have access to comprehensive internal documentation about all Aven products, services, and policies.

GUIDELINES:
- Search thoroughly using the most relevant keywords from the user's query
- Focus on official Aven information only
- If multiple relevant documents exist, prioritize the most current and comprehensive ones
- Never assume information - only use what's actually found in the knowledge base""",
        llm=llm,
        tools=[knowledge_base_search],
        verbose=True
    )
    
    search_agent = Agent(
        role='Web Research Specialist',
        goal='Find current, accurate information about Aven from official web sources',
        backstory="""You are a specialist in finding current information about Aven from official web sources. You only search when the knowledge base is insufficient for time-sensitive queries.

GUIDELINES:
- Only search for information that requires current/external verification
- Focus on official Aven sources and announcements
- Verify information credibility before including it
- If search fails or returns no results, acknowledge this clearly""",
        llm=llm,
        tools=[web_search],
        verbose=True
    )
    
    response_agent = Agent(
        role='Customer Support Assistant',
        goal='Generate helpful, accurate, and friendly responses following strict Aven guidelines',
        backstory=f"""You are Aven's official customer support AI assistant. You represent the Aven brand and must maintain the highest standards of accuracy and helpfulness.

{SYSTEM_PROMPT}

RESPONSE STYLE:
- Professional yet friendly tone
- Clear, concise explanations perfect for voice interaction
- Action-oriented guidance when possible
- Always include appropriate contact information when needed""",
        llm=llm,
        verbose=True
    )
    
    return router_agent, retrieval_agent, search_agent, response_agent

# Legacy crew-based function (optimized)
def create_tasks(query, router_agent, retrieval_agent, search_agent, response_agent):
    route_task = Task(
        description=f"""
        Analyze this user query about Aven: "{query}"
        
        Determine the best information source using these strict criteria:
        
        CHOOSE 'KNOWLEDGE_BASE' for:
        - Product information and features
        - Account-related questions  
        - How-to and troubleshooting
        - Policies and procedures
        - General Aven questions
        - Most customer support inquiries
        
        CHOOSE 'WEB_SEARCH' only for:
        - Queries explicitly asking for "current", "latest", "today", "recent" information
        - Time-sensitive regulatory or market changes
        - Breaking news about Aven
        
        Default to KNOWLEDGE_BASE unless there's a clear need for current external information.
        """,
        agent=router_agent,
        expected_output="Exactly one word: either 'KNOWLEDGE_BASE' or 'WEB_SEARCH'"
    )
    
    retrieve_task = Task(
        description=f"""
        Search Aven's knowledge base for: "{query}"
        
        Use the knowledge_base_search tool to find comprehensive information.
        
        REQUIREMENTS:
        - Use relevant keywords from the user's query
        - Return ALL relevant information found
        - If no relevant information exists, state this clearly
        - Do not add external information or assumptions
        """,
        agent=retrieval_agent,
        expected_output="Complete relevant information from knowledge base, or clear statement if no information found"
    )
    
    search_task = Task(
        description=f"""
        Find current web information about: "{query}"
        
        Use the web_search tool to find official, current information about Aven.
        
        REQUIREMENTS:
        - Only search if knowledge base is insufficient
        - Focus on official Aven sources
        - Summarize key findings clearly
        - If search fails, acknowledge and provide fallback guidance
        """,
        agent=search_agent,
        expected_output="Current information from official sources, or clear error message if search fails"
    )
    
    response_task = Task(
        description=f"""
        Create a comprehensive response to: "{query}"
        
        Use information from previous tasks following these guidelines:
        
        CONTENT REQUIREMENTS:
        - Base response entirely on gathered information
        - Prioritize knowledge base over web search results
        - Be accurate and never fabricate details
        - Include specific actionable steps when possible
        
        STYLE REQUIREMENTS:
        - Friendly and professional tone
        - Concise but complete (ideal for voice)
        - Clear structure and language
        - End with contact info if issue not fully resolved
        
        GUARDRAILS:
        - Only discuss Aven products and services
        - Never provide financial advice beyond official policies
        - Direct to support channels when appropriate
        - Acknowledge limitations honestly
        """,
        agent=response_agent,
        expected_output="A complete, accurate, and helpful response following all Aven brand guidelines"
    )
    
    return [route_task, retrieve_task, search_task, response_task]

# Main optimized function
def run_agent(query):
    """Main function - uses fast pipeline by default, falls back to crew if needed"""
    return run_agent_fast(query)

# Legacy crew function for complex scenarios
def run_agent_crew(query):
    """Legacy crew-based processing for complex multi-step queries"""
    if not llm:
        return "Agent system not available - missing LLM API keys. Please check your .env file."
    
    if not query or not isinstance(query, str) or len(query) < 3:
        logger.warning("Invalid query provided")
        return "Invalid query. Please provide a detailed question."
    
    # Create agents
    router_agent, retrieval_agent, search_agent, response_agent = create_agents()
    if not router_agent:
        return "Cannot create agents - missing required configuration."
    
    # Create tasks
    tasks = create_tasks(query, router_agent, retrieval_agent, search_agent, response_agent)
    
    # Create crew with concurrent processing where possible
    crew = Crew(
        agents=[router_agent, retrieval_agent, search_agent, response_agent],
        tasks=tasks,
        verbose=1,
        process=Process.sequential  # Can be changed to hierarchical for better parallelism
    )
    
    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        # Fallback to fast pipeline
        return run_agent_fast(query)

if __name__ == '__main__':
    if not llm:
        print("Cannot run agent - missing required API keys. Please set up your .env file.")
    else:
        print("Aven Support Agent (Optimized for Voice)")
        print("Type 'quit' to exit")
        
        while True:
            query = input('\nEnter your question: ')
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            start_time = time.time()
            response = run_agent(query)
            end_time = time.time()
            
            print(f"\nResponse ({end_time - start_time:.2f}s): {response}")