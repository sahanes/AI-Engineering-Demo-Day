"""
Server script for the MentalHealthCounselingBot-server.
This launches the FastMCP server.
"""

import os
import re
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
import logging

from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from MentalMindBot_AI_RAG_with_Agent_with_Memory import create_qa_chain, parent_document_retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_test")
# Load environment variables
load_dotenv()

# Initialize the MCP server
mcp = FastMCP("MentalHealthCounselingBot-server")

# Initialize the QA chain
parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)

@mcp.tool()
async def rag_tool(question: str) -> str:
    """Use RAG to find relevant mental health information from the knowledge base.
    
    Args:
        question: The mental health related question to search for
        
    Returns:
        str: Relevant information from the knowledge base
    """
    try:
        if isinstance(question, list) and len(question) == 2:
            actual_query = question[0]
            if isinstance(actual_query, list):
                actual_query = actual_query[0]
            response = await parent_document_retriever_qa_chain.ainvoke({"question": actual_query})
            return response["response"]
        
        response = await parent_document_retriever_qa_chain.ainvoke({"question": question})
        return response["response"]
    except Exception as e:
        logger.error(f"Error in rag_tool: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def pubmed_search(query: str) -> str:
    """Search PubMed for mental health consultation and counseling research."""
    try:
        pubmed_tool = PubmedQueryRun()
        result = await pubmed_tool.ainvoke({"query": query})
        return result
    except Exception as e:
        logger.error(f"Error in pubmed_search: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def arxiv_search(query: str) -> str:
    """Search Arxiv for mental health research and counseling literature."""
    try:
        arxiv_tool = ArxivQueryRun()
        result = await arxiv_tool.ainvoke({"query": query})
        return result
    except Exception as e:
        logger.error(f"Error in arxiv_search: {str(e)}")
        return f"Error: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """Search trusted mental health resources and websites for evidence-based information and guidance."""
    try:
        duckduckgo_tool = DuckDuckGoSearchRun()
        result = await duckduckgo_tool.ainvoke({"query": query})
        return result
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting MCP server...")
    try:
        mcp.run(transport="stdio")
        logger.info("MCP server running")
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")