# Multiagent System for Aven Customer Support

A scalable AI Customer Support Agent built with Python 3.10 and CrewAI, designed to provide fast, accurate, and friendly responses about Aven through voice or text chat.

## Features

- **Multi-Agent Architecture**: Built with CrewAI for scalable, autonomous AI agents
- **RAG-Powered Knowledge Base**: Retrieval-Augmented Generation using Pinecone vector database
- **Fallback Mechanisms**: Web search with SerpAPI when knowledge base is insufficient
- **Voice & Text Support**: Integrated with Vapi for seamless voice/text interactions
- **Guardrails**: Built-in safety checks for sensitive content
- **Multi-LLM Support**: Works with OpenAI GPT or Google Gemini models
- **Modern Frontend**: Next.js web interface with TypeScript and Tailwind CSS

## Setup

### Prerequisites
- Python 3.10+ (required for CrewAI)
- Node.js 18+ (for frontend)
- API Keys for: OpenAI/Gemini, Pinecone, SerpAPI, Vapi

### Installation

1. Clone the repository and navigate to the project directory

2. **Backend Setup**:
   ```bash
   cd backend
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

### Quick Start

1. **Data Ingestion**: Scrape and index Aven's support content
   ```bash
   cd backend
   source venv/bin/activate
   python ingest.py
   ```

2. **Test RAG Retrieval**: Query the knowledge base
   ```bash
   python rag_query.py
   ```

3. **Run Multi-Agent System**: Start the full agentic workflow
   ```bash
   python agent.py
   ```

4. **Run Frontend**: Start the Next.js development server
   ```bash
   cd frontend
   npm run dev
   ```

## Project Structure

```
aven-support-bot/
├── backend/              # Python backend with CrewAI
│   ├── venv/            # Virtual environment (Python 3.10)
│   ├── requirements.txt # Python dependencies with CrewAI
│   ├── env.example      # Environment variables template
│   ├── ingest.py        # Data ingestion pipeline
│   ├── rag_query.py     # RAG testing script
│   └── agent.py         # Multi-agent CrewAI system
├── frontend/            # Next.js frontend application
│   ├── src/            # Source code with TypeScript
│   ├── public/         # Static assets
│   ├── package.json    # Node.js dependencies
│   └── ...             # Next.js configuration files
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Development Status

✅ **Completed:**
- Python 3.10 environment setup
- CrewAI installation and configuration
- Data ingestion pipeline (Pinecone + LangChain)
- Basic RAG query system
- Multi-agent architecture foundation

🚧 **Next Steps:**
- Web interface with Vapi integration
- Evaluation framework with test questions
- Meeting scheduling tool integration
- Production deployment setup

## Architecture Overview

The system uses a multi-agent approach with CrewAI:

1. **Guardrails Agent**: Checks for sensitive content
2. **Router Agent**: Determines query routing strategy  
3. **Retrieval Agent**: Searches knowledge base (RAG)
4. **Search Agent**: Fallback web search via SerpAPI
5. **Response Agent**: Generates friendly, cited responses

## Contributing

This project is designed to be scalable and extensible. Add new agents, tools, or integrations as needed for your specific use case. 
