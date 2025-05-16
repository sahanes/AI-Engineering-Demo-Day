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

# The Matrix-style appearance comes from several key files
## globals.css - Contains all the Matrix styling:
  - Green color scheme (--matrix-green: #00ff41)
  - Terminal-like font (JetBrains Mono)
  - Matrix rain animation
  - Glowing effects
  - Terminal-style input and messages
## layout.tsx - Sets up the basic structure
  - Imports the Matrix font
  - Creates the terminal-like container
  - Adds the "NEURAL HEALTH INTERFACE" header
## page.tsx - Implements the terminal interface:
  - Message rendering with different styles for user/AI/system
  - Terminal-style input with "QUERY>" prefix
  - Loading animations and effects
  
#### Key Matrix-style features
  - Green text on black background
  - Terminal-like interface
  - Glowing effects
  - Matrix rain animation in the background
  - Terminal-style prefixes ("> USER:", "> NEURAL:", "> SYS:")
  - Blinking cursor effects
  - Loading dots animation
The styling is achieved through a combination of:
CSS variables for colors
CSS animations for effects
Tailwind CSS for layout
Custom CSS classes for Matrix-specific styling
Let me know if you want to modify any specific part of the Matrix styling!
