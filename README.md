# Mental Health Counseling Bot

A mental health counseling bot that uses advanced language models and retrieval-augmented generation (RAG) to provide evidence-based mental health information and support.

## Features

- Integration with PubMed for medical research
- RAG-based knowledge retrieval
- Real-time web search capabilities
- Integration with Global MCP tools

## Installation

```bash
# Install dependencies
uv pip install -e .

# Start the server
uv run server.py
```

## Environment Variables

Create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
python -m chainlit run Bot_Integration_with_MCP_Tools.py
``` 