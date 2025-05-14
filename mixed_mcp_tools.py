"""
Mixed MCP Tools Module

This module provides a mix of local and global MCP tools,
with proper fallback to local implementations when global ones fail.
"""

import subprocess
import json
import time
import threading
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain.tools import Tool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mixed_mcp_tools")

# Globals
DEFAULT_TIMEOUT = 120  # Increased timeout for tools to complete

def ensure_server_running():
    """Ensure the local MCP server is running - platform independent"""
    try:
        # Use platform-independent subprocess check
        try:
            # Unix-like systems
            result = subprocess.run(["pgrep", "-f", "uv run server.py"], 
                                   capture_output=True, text=True)
            is_running = result.returncode == 0
        except FileNotFoundError:
            # Windows or systems without pgrep
            result = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
            is_running = "uv run server.py" in result.stdout
            
        if is_running:
            logger.info("MCP server is already running")
        return True
            
        logger.info("Starting MCP server...")
        # Use shell=True on Windows systems for better process creation
        is_windows = os.name == 'nt'
        process = subprocess.Popen(
            ["uv", "run", "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=is_windows,
            start_new_session=True
        )
        time.sleep(10)  # Extended wait time for server initialization
        if process.poll() is not None:
            logger.error(f"Failed to start MCP server. Error: {process.stderr.read() if process.stderr else 'Unknown error'}")
            return False
        logger.info("MCP server started successfully")
        return True
    except Exception as e:
        logger.error(f"Error checking/starting server: {e}")
        # Return True anyway to allow other tools to work
        return True

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=DEFAULT_TIMEOUT):
    """Run a function with a timeout"""
    result = [None]
    error = [None]
    completed = [False]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
            completed[0] = True
        except Exception as e:
            error[0] = str(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        return f"Timeout after {timeout_duration} seconds"
    if error[0]:
        return f"Error: {error[0]}"
    if completed[0]:
        return result[0]
    return "Unknown error occurred"

# Direct implementations of tools (no MCP server needed)
def direct_pubmed_search(query_str):
    """Direct implementation of PubMed search"""
    try:
        logger.info(f"Running direct PubMed search for query: {query_str}")
        pubmed_tool = PubmedQueryRun()
        result = pubmed_tool.run(query_str)
        logger.info("PubMed search completed successfully")
        return result
    except Exception as e:
        logger.error(f"PubMed search failed: {str(e)}")
        return f"Error in PubMed search: {str(e)}"

def direct_arxiv_search(query_str):
    """Direct implementation of ArXiv search"""
    try:
        logger.info(f"Running direct ArXiv search for query: {query_str}")
        arxiv_tool = ArxivQueryRun()
        result = arxiv_tool.run(query_str)
        logger.info("ArXiv search completed successfully")
        return result
    except Exception as e:
        logger.error(f"ArXiv search failed: {str(e)}")
        return f"Error in ArXiv search: {str(e)}"

def direct_web_search(query_str):
    """Direct implementation of web search using DuckDuckGo"""
    try:
        logger.info(f"Running direct web search for query: {query_str}")
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query_str)
        logger.info("Web search completed successfully")
        return result
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return f"Error in web search: {str(e)}"

def direct_git_info(query_str):
    """Direct implementation of git info"""
    try:
        logger.info(f"Running direct git info for query: {query_str}")
        if "status" in query_str.lower():
            git_cmd = ["git", "status"]
        elif "log" in query_str.lower():
            git_cmd = ["git", "log", "--oneline", "-n", "5"]
        elif "branch" in query_str.lower():
            git_cmd = ["git", "branch"]
        else:
            git_cmd = ["git", "log", "--oneline", "-n", "5"]
            
        result = subprocess.run(git_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Git info completed successfully")
            return result.stdout
        logger.error(f"Git command failed: {result.stderr}")
        return f"Git command failed: {result.stderr}"
    except Exception as e:
        logger.error(f"Git info failed: {str(e)}")
        return f"Error in git info: {str(e)}"

def create_mixed_tool_belt():
    """Create a tool belt with direct implementations of all tools"""
    # Try to ensure server is running, but use direct implementations regardless
    ensure_server_running()
    
    # Import direct RAG implementation
    from direct_rag import direct_rag_query
    
    # Create tools with direct implementations
    pubmed_search = Tool(
        name="pubmed_search",
        description="Search PubMed for medical research articles",
        func=direct_pubmed_search
    )
    
    arxiv_search = Tool(
        name="arxiv_search",
        description="Search Arxiv for academic papers",
        func=direct_arxiv_search
    )
    
    # RAG tool uses direct implementation
    rag_tool = Tool(
        name="rag_tool",
        description="Search the knowledge base for mental health information",
        func=direct_rag_query
    )
    
    # Web search uses direct DuckDuckGo implementation
    web_search_tool = Tool(
        name="web_search",
        description="Search the web for information using DuckDuckGo",
        func=direct_web_search
    )
    
    # Git info uses local git commands directly
    git_info_tool = Tool(
        name="git_info",
        description="Get information about a git repository using local git commands",
        func=direct_git_info
    )
    
    # Return tool belt
    return [pubmed_search, arxiv_search, rag_tool, web_search_tool, git_info_tool]

# Additional helper function for better parsing of tool calls
def parse_tool_args(last_message):
    """Parse tool arguments from message more robustly"""
    try:
        if "function_call" not in last_message.additional_kwargs:
            return None, None
            
        function_call = last_message.additional_kwargs["function_call"]
        name = function_call.get("name", "")
        arguments = function_call.get("arguments", "{}")
        
        # Try to parse arguments
        try:
            args_dict = json.loads(arguments)
            # Handle both formats
            if "args" in args_dict and isinstance(args_dict["args"], list) and args_dict["args"]:
                query = args_dict["args"][0]
            elif "query" in args_dict:
                query = args_dict["query"]
            else:
                query = arguments  # Just use the raw string
        except:
            query = arguments
            
        return name, query
    except Exception as e:
        print(f"Error parsing tool args: {e}")
        return None, None

# Test function
def test_tools():
    """Test the mixed MCP tools"""
    tools = create_mixed_tool_belt()
    
    print("Testing tools:")
    test_queries = {
        "pubmed_search": "anxiety and depression treatment",
        "arxiv_search": "mental health machine learning",
        "rag_tool": "cognitive behavioral therapy for anxiety",
        "web_search": "latest mental health research findings",
        "git_info": "status"
    }
    
    for tool in tools:
        print(f"\nTesting {tool.name}...")
        try:
            query = test_queries.get(tool.name, "mental health")
            result = tool.func(query)
            print(f"Result: {result[:200]}..." if result and len(result) > 200 else result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_tools()




