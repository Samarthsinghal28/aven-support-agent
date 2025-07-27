import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
import logging
import openai
import numpy as np
import re
import json
import csv
from tqdm import tqdm

# --- Configuration ---
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DATASET_PATH = "eval_dataset.csv"
RESULTS_PATH = "evaluation_results.csv"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Setup OpenAI Client ---
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized.")
except Exception as e:
    logger.error(f"OpenAI API key not found or invalid: {e}")
    exit(1)

def extract_questions_from_markdown(markdown_path):
    """Extract questions and expected answers from the markdown file"""
    with open(markdown_path, 'r') as f:
        content = f.read()
    
    # Regular expression to extract question-answer pairs
    pattern = r'\d+\.\s+\*\*Question:\*\*\s+(.*?)\n\s+\*\*Answer:\*\*\s+(.*?)(?=\n\d+\.\s+\*\*Question:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    # Create a dataframe
    questions = []
    answers = []
    for q, a in matches:
        questions.append(q.strip())
        answers.append(a.strip())
    
    # Save to CSV
    df = pd.DataFrame({
        'question': questions,
        'answer': answers
    })
    
    df.to_csv(DATASET_PATH, index=False)
    logger.info(f"Extracted {len(df)} questions and answers to {DATASET_PATH}")
    return df

def query_agent(question):
    """Send a question to the agent and get the response"""
    try:
        url = f"{BACKEND_URL}/chat"
        payload = {
            "message": question,
            "session_id": f"eval_{int(time.time())}"
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            logger.error(f"Error querying agent: {response.status_code} - {response.text}")
            return "[AGENT_ERROR]"
    except Exception as e:
        logger.error(f"Exception when querying agent: {e}")
        return "[AGENT_ERROR]"

def get_openai_embedding(text, model=OPENAI_EMBEDDING_MODEL):
    """Get embeddings from OpenAI API"""
    if not text or text == "[AGENT_ERROR]":
        return None
    
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def calculate_cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    if v1 is None or v2 is None:
        return 0.0
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

def evaluate_similarity(agent_answer, ground_truth):
    """Evaluate similarity between agent answer and ground truth using embeddings"""
    embedding_agent = get_openai_embedding(agent_answer)
    embedding_truth = get_openai_embedding(ground_truth)
    
    if embedding_agent and embedding_truth:
        return calculate_cosine_similarity(embedding_agent, embedding_truth)
    return 0.0

def format_results_markdown(results):
    """Format results as markdown for easy viewing"""
    markdown = "# Agent Evaluation Results\n\n"
    markdown += "## Summary\n\n"
    
    # Calculate average similarity
    similarities = [r['similarity'] for r in results if r['similarity'] > 0]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    markdown += f"- **Total Questions**: {len(results)}\n"
    markdown += f"- **Average Similarity Score**: {avg_similarity:.4f}\n"
    markdown += f"- **Questions with Similarity ≥ 0.80**: {sum(1 for r in results if r['similarity'] >= 0.80)}\n"
    markdown += f"- **Questions with Similarity ≥ 0.90**: {sum(1 for r in results if r['similarity'] >= 0.90)}\n\n"
    
    markdown += "## Detailed Results\n\n"
    
    for i, result in enumerate(results, 1):
        markdown += f"### Question {i}: {result['question']}\n\n"
        markdown += f"**Expected Answer:**\n{result['ground_truth']}\n\n"
        markdown += f"**Agent Answer:**\n{result['agent_answer']}\n\n"
        markdown += f"**Similarity Score:** {result['similarity']:.4f}\n\n"
        markdown += "---\n\n"
    
    return markdown

def main():
    """Main evaluation function"""
    # Check if dataset exists, if not, extract from markdown
    if os.path.exists(DATASET_PATH):
        logger.info(f"Loading existing dataset from {DATASET_PATH}")
        dataset = pd.read_csv(DATASET_PATH)
    else:
        logger.info("Dataset not found, extracting from markdown")
        dataset = extract_questions_from_markdown("eval-set.md")
    
    # Check if the backend is available
    try:
        health_check = requests.get(f"{BACKEND_URL}/health")
        if health_check.status_code != 200:
            logger.error(f"Backend health check failed: {health_check.status_code}")
            return
        logger.info("Backend health check passed")
    except Exception as e:
        logger.error(f"Backend not available: {e}")
        return
    
    # Initialize results
    results = []
    total_similarity = 0.0
    successful_queries = 0
    
    # Process each question
    for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Evaluating questions"):
        question = row['question']
        ground_truth = row['answer']
        
        logger.info(f"Processing question {i+1}/{len(dataset)}")
        
        # Query the agent
        agent_answer = query_agent(question)
        
        # Calculate similarity
        similarity = 0.0
        if agent_answer != "[AGENT_ERROR]" and isinstance(ground_truth, str):
            similarity = evaluate_similarity(agent_answer, ground_truth)
            total_similarity += similarity
            successful_queries += 1
        
        # Store results
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'agent_answer': agent_answer,
            'similarity': similarity
        })
        
        # Add a small delay to avoid rate limits
        time.sleep(1)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate average similarity
    avg_similarity = total_similarity / successful_queries if successful_queries > 0 else 0
    
    # Save results
    results_df.to_csv(RESULTS_PATH, index=False)
    
    # Generate markdown report
    markdown_report = format_results_markdown(results)
    with open("evaluation_results.md", "w") as f:
        f.write(markdown_report)
    
    # Print summary
    logger.info(f"\nEvaluation complete!")
    logger.info(f"Average similarity score: {avg_similarity:.4f}")
    logger.info(f"Questions with similarity >= 0.80: {sum(1 for r in results if r['similarity'] >= 0.80)} / {len(results)}")
    logger.info(f"Questions with similarity >= 0.90: {sum(1 for r in results if r['similarity'] >= 0.90)} / {len(results)}")
    logger.info(f"Results saved to {RESULTS_PATH} and evaluation_results.md")

if __name__ == "__main__":
    main() 