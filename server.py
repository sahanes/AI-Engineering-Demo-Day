# """
# Server script for the MentalHealthCounselingBot-server.
# This launches the FastMCP server with improved JSON-RPC handling.
# """

# import os
# import re
# import requests
# import signal
# import sys
# import asyncio
# import json
# import logging
# import time
# import subprocess
# from typing_extensions import Optional, Dict, Any, List
# from dotenv import load_dotenv

# # Import your local tool dependencies
# from fastmcp import FastMCP
# from langchain_community.tools.pubmed.tool import PubmedQueryRun
# from langchain_community.tools.arxiv.tool import ArxivQueryRun
# # from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
# from tavily import TavilyClient

# # Import your RAG components
# from Bot_Integration_with_MCP_Tools import create_qa_chain, parent_document_retriever

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG for more details
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     stream=sys.stdout
# )
# logger = logging.getLogger("mcp_server")

# # Load environment variables
# try:
#     load_dotenv()
#     logger.info("Environment variables loaded")
# except Exception as e:
#     logger.warning(f"Error loading environment variables: {e}")

# # Initialize the MCP server
# mcp = FastMCP("MentalHealthCounselingBot-server")
# logger.info("FastMCP server initialized")

# # Initialize tools with error handlingclient = TavilyClient(os.getenv("TAVILY_API_KEY"))
# pubmed_tool = PubmedQueryRun()
# arxiv_tool = ArxivQueryRun()
# # duckduckgo_tool = DuckDuckGoSearchRun()
# logger.info("Search tools initialized")
    
# # Initialize the QA chain if parent_document_retriever is available
# if parent_document_retriever and hasattr(parent_document_retriever, 'add_documents'):
#     try:
#         # Don't load documents at startup - they'll be loaded when needed
#         parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)
#         logger.info("QA chain initialized")
#     except Exception as e:
#         logger.warning(f"Error initializing QA chain: {e}")
#         # Create a fallback QA chain
#         async def parent_document_retriever_qa_chain_invoke(*args, **kwargs):
#             return {"response": "RAG functionality unavailable due to initialization error."}
#         parent_document_retriever_qa_chain = type('obj', (object,), {'ainvoke': parent_document_retriever_qa_chain_invoke})()
# else:
#     logger.warning("parent_document_retriever not available, skipping QA chain initialization")
#     # Placeholder for the QA chain
#     async def parent_document_retriever_qa_chain_invoke(*args, **kwargs):
#         return {"response": "RAG functionality unavailable."}
#     parent_document_retriever_qa_chain = type('obj', (object,), {'ainvoke': parent_document_retriever_qa_chain_invoke})()

# # Initialize this once at the top level
# tavily_client = None
# try:
#     tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
#     logger.info("Tavily client initialized")
# except Exception as e:
#     logger.warning(f"Error initializing Tavily client: {e}")

# # Register tools with FastMCP
# @mcp.tool()
# async def rag_tool(query: str) -> str:
#     """Search the knowledge base for mental health information"""
#     # FIXED: Pass query in a dictionary with "question" key
#     result = await parent_document_retriever_qa_chain.ainvoke({"question": query})
#     return result.get("response", "No response from RAG tool")

# @mcp.tool()
# async def pubmed_search(query: str) -> str:
#     """Search PubMed for medical research articles"""
#     try:
#         # Handle both async and sync implementations
#         if hasattr(pubmed_tool, 'arun'):
#             return await pubmed_tool.arun(query)
#         return pubmed_tool.run(query)
#     except Exception as e:
#         logger.error(f"Error in pubmed_search: {e}")
#         return f"Error searching PubMed: {str(e)}"

# @mcp.tool()
# async def arxiv_search(query: str) -> str:
#     """Search Arxiv for academic papers"""
#     return await arxiv_tool.arun(query)

# @mcp.tool()
# def web_search(query: str) -> str:
#     """Search the web for information about the given query"""
#     search_results = tavily_client.get_search_context(query=query)
#     return search_results

# @mcp.tool()
# def git_info(repo_path: str = ".", info_type: str = "status") -> str:
#     """Get information about a git repository."""
#     try:
#         if info_type == "status":
#             result = subprocess.check_output(["git", "-C", repo_path, "status"], text=True)
#         elif info_type == "log":
#             result = subprocess.check_output(["git", "-C", repo_path, "log", "--oneline", "-5"], text=True)
#         else:
#             result = f"Unknown info_type: {info_type}"
#         return result
#     except Exception as e:
#         return f"Error: {e}"

# if __name__ == "__main__":
#     # Start the MCP server with stdio transport
#     mcp.run(transport="stdio")
#     print("Server ready to process requests via stdio")
"""
Server script for the MentalHealthCounselingBot-server.
This launches the FastMCP server with improved JSON-RPC handling.
"""

import os
import re
import requests
import signal
import sys
import asyncio
import json
import logging
import time
import subprocess
from typing_extensions import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import your local tool dependencies
from fastmcp import FastMCP
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from tavily import TavilyClient

# Import your RAG components
from Bot_Integration_with_MCP_Tools import create_qa_chain, parent_document_retriever

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("mcp_server")

# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.warning(f"Error loading environment variables: {e}")

# Initialize the MCP server
mcp = FastMCP("MentalHealthCounselingBot-server")
logger.info("FastMCP server initialized")

# Initialize tools with error handlingclient = TavilyClient(os.getenv("TAVILY_API_KEY"))
pubmed_tool = PubmedQueryRun()
arxiv_tool = ArxivQueryRun()
duckduckgo_tool = DuckDuckGoSearchRun()
logger.info("Search tools initialized")
    
# Initialize the QA chain if parent_document_retriever is available
if parent_document_retriever and hasattr(parent_document_retriever, 'add_documents'):
    try:
        # Don't load documents at startup - they'll be loaded when needed
        parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)
        logger.info("QA chain initialized")
    except Exception as e:
        logger.warning(f"Error initializing QA chain: {e}")
        # Create a fallback QA chain
        async def parent_document_retriever_qa_chain_invoke(*args, **kwargs):
            return {"response": "RAG functionality unavailable due to initialization error."}
        parent_document_retriever_qa_chain = type('obj', (object,), {'ainvoke': parent_document_retriever_qa_chain_invoke})()
else:
    logger.warning("parent_document_retriever not available, skipping QA chain initialization")
    # Placeholder for the QA chain
    async def parent_document_retriever_qa_chain_invoke(*args, **kwargs):
        return {"response": "RAG functionality unavailable."}
    parent_document_retriever_qa_chain = type('obj', (object,), {'ainvoke': parent_document_retriever_qa_chain_invoke})()

# Register tools with FastMCP
@mcp.tool()
async def rag_tool(query: str) -> str:
    """Search the knowledge base for mental health information"""
    result = await parent_document_retriever_qa_chain.ainvoke(query)
    return result.get("response", "No response from RAG tool")

@mcp.tool()
async def pubmed_search(query: str) -> str:
    """Search PubMed for medical research articles"""
    return await pubmed_tool.arun(query)

@mcp.tool()
async def arxiv_search(query: str) -> str:
    """Search Arxiv for academic papers"""
    return await arxiv_tool.arun(query)

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def git_info(repo_path: str = ".", info_type: str = "status") -> str:
    """Get information about a git repository."""
    try:
        if info_type == "status":
            result = subprocess.check_output(["git", "-C", repo_path, "status"], text=True)
        elif info_type == "log":
            result = subprocess.check_output(["git", "-C", repo_path, "log", "--oneline", "-5"], text=True)
        else:
            result = f"Unknown info_type: {info_type}"
        return result
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Start the MCP server with stdio transport
    mcp.run(transport="stdio")
    print("Server ready to process requests via stdio")