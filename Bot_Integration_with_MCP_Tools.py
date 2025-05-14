from langchain.schema import SystemMessage
import uuid
from langchain_core.utils.function_calling import convert_to_openai_function
from mixed_mcp_tools import create_mixed_tool_belt, parse_tool_args
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv
from langchain_community.document_loaders import ArxivLoader
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from langchain_openai import OpenAIEmbeddings
from langchain_core.stores import InMemoryStore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from IPython.display import display, Markdown
from langgraph.prebuilt import ToolNode

# Updated imports for LangGraph 0.4.3
from typing_extensions import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, END

# We don't need langgraph.toolkit - we'll create a simple tool executor
from langgraph.checkpoint.memory import MemorySaver

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    SystemMessage,
    AIMessage,
    HumanMessage
)
import operator
import json, asyncio
import logging
import sys
import time
import os
import httpx
import subprocess
import re

# FastAPI imports for the API endpoint
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("mental_health_bot")

"""
This function will load our environment file (.env) if it is present.
"""
load_dotenv()

# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN.

try:
    docs = ArxivLoader(
        query='"mental health counseling" AND (data OR analytics OR "machine learning")',
        load_max_docs=2,
        sort_by="submittedDate",
        sort_order="descending",
        load_all_available_meta=True,  # Load metadata without PDFs
        download_pdf=False  # Skip PDF downloads
    ).load()
except Exception as e:
    print(f"Warning: Error loading Arxiv documents: {e}")
    docs = []  # Use empty list if loading fails


### 2. CREATE QDRANT CLIENT VECTORE STORE

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="split_parents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Initialize embeddings with SSL verification settings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    request_timeout=30,
    max_retries=3,
    http_client=httpx.Client(verify=False)  # Disable SSL verification temporarily
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="split_parents",
    embedding=embeddings,
)

store = InMemoryStore()

### 3. CREATE PARENT DOCUMENT TEXT SPLITTER AND RETRIEVER INITIATED
parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
parent_document_retriever.add_documents(docs)

### 4. CREATE PROMPT OBJECT
RAG_PROMPT = """\
Your are a professional mental health advisor. Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

### 5. CREATE CHAIN PIPELINE RETRIEVER

openai_chat_model = ChatOpenAI(
    model="gpt-4.1-mini",
    streaming=True,
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    request_timeout=30,
    max_retries=3,
    http_client=httpx.Client(verify=False)  # Disable SSL verification temporarily
)


def create_qa_chain(retriever):
    mentahealth_qa_llm = openai_chat_model

    created_qa_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {
            "response": rag_prompt | mentahealth_qa_llm | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )
    return created_qa_chain


### 6. DEFINE LIST OF TOOLS AVAILABLE FOR AND TOOL EXECUTOR WRAPPED AROUND THEM
# Create tool belt with MCP tools
tool_belt = create_mixed_tool_belt()

# Simple tool executor that doesn't rely on langgraph.toolkit
class SimpleToolExecutor:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
        
    async def ainvoke(self, tool_invocation):
        """Invoke a tool asynchronously."""
        tool_name = tool_invocation.get("tool") or tool_invocation.get("name")
        tool_input = tool_invocation.get("tool_input") or tool_invocation.get("input")
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
            
        tool = self.tools[tool_name]
        
        try:
            # Handle invocation based on whether the tool is async or not
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(tool_input)
            elif hasattr(tool, "invoke"):
                return tool.invoke(tool_input)
            elif callable(tool.func):  # Use .func for LangChain tools
                # Simple callable
                result = tool.func(tool_input)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            else:
                return f"Error: Tool '{tool_name}' is not callable"
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"

# Initialize our simple tool executor with the updated tool belt
tool_executor = SimpleToolExecutor(tool_belt)

# Initialize the model
model = ChatOpenAI(
    model="gpt-3.5-turbo",  # Using a model that supports function calling
    temperature=0,
    streaming=True
)
model = model.with_config({"callbacks": [StreamingStdOutCallbackHandler()]})

# Convert tools to OpenAI function format
functions = [convert_to_openai_function(t) for t in tool_belt]

# Bind tools to the model
model = model.bind_functions(functions)
model = model.with_config(tags=["final_node"])

### 8. USING the TypedDict FROM THE typing module AND THE langchain_core.messages module, A CUSTOM TYPE NAMED AgentState CREATED.
# THE AgentState type HAS A FIELD NAMED <messages> THAT IS OF TYPE Annotated[Sequence[BaseMessage], operator.add].
# Sequence[BaseMessage]: INDICATES THAT MESSAGES ARE A SEQUENCE OF BaseMessage OBJECTS.
# Annotated: USED TO ATTACH MEATADATA TO THE TYPE, THEN THE MESSAGE FIELD TREATED AS CONCATENABLE SEQUENCE OF BASEMASSAGES TO OPERATOR.ADD FUNCTION.


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


### SIMPLIFIED CALL MODEL FUNCTION TO AVOID INFINITE LOOPS
async def call_model(state):
    # Absolute maximum message count to prevent infinite loops
    if len(state["messages"]) > 20:
        return {
            "messages": [
                AIMessage(content="I've gathered information from multiple sources. Let me summarize what I've found...")
            ]
        }
    
    messages = state["messages"]
    
    try:
        # Get list of tools already used and count their usage
        tool_usage = {}
        for msg in state["messages"]:
            if hasattr(msg, "name") and msg.name in ["pubmed_search", "rag_tool", "web_search"]:
                tool_usage[msg.name] = tool_usage.get(msg.name, 0) + 1
        
        # Define available tools in order of preference
        available_tools = ["pubmed_search", "rag_tool", "web_search"]
        
        # Find tools that haven't exceeded their retry limit (max 3 attempts per tool)
        usable_tools = [tool for tool in available_tools if tool_usage.get(tool, 0) < 3]
        
        if not usable_tools:
            # If all tools have been tried too many times, provide a final response
            response = await model.ainvoke(
                messages + [
                    SystemMessage(content="Provide a comprehensive response based on the information gathered so far.")
                ]
            )
            return {"messages": [response]}
        
        # Choose the next tool that hasn't been tried too many times
        next_tool = usable_tools[0]
        
        # Create a message to guide the model
        system_message = SystemMessage(
            content=f"""Use the {next_tool} tool to search for information. 
            If this tool doesn't provide useful information within 10 seconds, we'll try another tool.
            Make sure to use the tool's response in your final answer."""
        )
        
        # Get response from model
        response = await model.ainvoke(messages + [system_message])
        
        # Ensure we have some content
        if not response.content:
            response.content = f"Let me search for information about that using {next_tool}."
        
        return {"messages": [response]}
            
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return {
            "messages": [
                AIMessage(content="I apologize, but I'm experiencing some technical difficulties. Let me provide a general response based on my knowledge.")
            ]
        }

### SIMPLIFIED CALL TOOL FUNCTION TO AVOID INFINITE LOOPS
async def call_tool(state):
    last_message = state["messages"][-1]
    
    try:
        # Check if message has a function call
        if not hasattr(last_message, "additional_kwargs") or "function_call" not in last_message.additional_kwargs:
            logger.warning("No function call found in message")
            return {
                "messages": [
                    FunctionMessage(
                        content="No tool to call. Let's try another approach.",
                        name="system"
                    )
                ]
            }
        
        # Get the function call
        function_call = last_message.additional_kwargs["function_call"]
        tool_name = function_call["name"]
        logger.info(f"Attempting to call tool: {tool_name}")
        
        # Parse the arguments
        try:
            args = json.loads(function_call["arguments"])
            tool_input = args.get("__arg1", state["messages"][0].content)
        except json.JSONDecodeError:
            tool_input = state["messages"][0].content
        
        # Log what we're doing
        logger.info(f"Calling tool {tool_name} with input: {tool_input}")
        
        # Get the tool directly from the tool belt
        tool = None
        for t in tool_belt:
            if t.name == tool_name:
                tool = t
                break
                
        if not tool:
            logger.error(f"Tool {tool_name} not found in tool belt")
            return {
                "messages": [
                    FunctionMessage(
                        content=f"I couldn't find the {tool_name} tool. Let me try a different approach.",
                        name="error"
                    )
                ]
            }
            
        # Execute the tool directly
        try:
            logger.info(f"Executing {tool_name} directly with input: {tool_input[:100]}...")
            start_time = time.time()
            result = tool.func(tool_input)
            elapsed_time = time.time() - start_time
            logger.info(f"Tool {tool_name} completed in {elapsed_time:.2f} seconds")
            
            if result and str(result).strip():
                return {
                    "messages": [
                        FunctionMessage(content=str(result), name=tool_name)
                    ]
                }
            else:
                logger.warning(f"Tool {tool_name} returned empty result")
                return {
                    "messages": [
                        FunctionMessage(
                            content=f"I couldn't find any relevant information using {tool_name}. Let me try a different approach.",
                            name="error"
                        )
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {str(e)}")
            return {
                "messages": [
                    FunctionMessage(
                        content=f"Error with {tool_name}: {str(e)}. Let me try a different approach.",
                        name="error"
                    )
                ]
            }
        
    except Exception as e:
        logger.error(f"Error in call_tool: {e}")
        return {
            "messages": [
                FunctionMessage(
                    content="An error occurred. Let's try a different approach.",
                    name="error"
                )
            ]
        }

### SIMPLIFIED SHOULD_CONTINUE FUNCTION WITH STRICT MESSAGE COUNT LIMIT
def should_continue(state):
    # Hard limit on message count to prevent infinite loops
    if len(state["messages"]) > 15:
        logger.warning("Maximum message count reached, ending conversation")
        return "end"
    
    # Get list of tools already used and count their usage
    tool_usage = {}
    for msg in state["messages"]:
        if hasattr(msg, "name") and msg.name in ["pubmed_search", "rag_tool", "web_search"]:
            tool_usage[msg.name] = tool_usage.get(msg.name, 0) + 1
    
    # Check if any tool has been used too many times
    if any(count >= 5 for count in tool_usage.values()):
        logger.info("Some tools have been used too many times, moving to final response")
        return "end"
    
    # Count consecutive tool calls without meaningful progress
    tool_call_count = 0
    for msg in reversed(state["messages"]):
        if (hasattr(msg, "additional_kwargs") and 
            "function_call" in msg.additional_kwargs):
            tool_call_count += 1
        else:
            break
    
    # If we've had too many consecutive tool calls without progress, end
    if tool_call_count >= 5:
        logger.warning("Too many consecutive tool calls without progress, ending conversation")
        return "end"
    
    # Count consecutive errors
    error_count = 0
    for msg in reversed(state["messages"]):
        if hasattr(msg, "name") and msg.name in ["error", "timeout"]:
            error_count += 1
        else:
            break
    
    # If we've had too many consecutive errors, end the conversation
    if error_count >= 3:
        logger.warning("Too many consecutive errors, ending conversation")
        return "end"
    
    # Check if the last message is an AI message with no function call
    last_message = state["messages"][-1]
    is_final_response = (
        isinstance(last_message, AIMessage) and
        hasattr(last_message, "content") and
        last_message.content and
        (not hasattr(last_message, "additional_kwargs") or
         "function_call" not in last_message.additional_kwargs)
    )
    
    # If this looks like a final response, end
    if is_final_response:
        logger.info("Received final response, ending conversation")
        return "end"
    
    # If we've used all tools successfully, end
    if all(tool in tool_usage for tool in ["pubmed_search", "rag_tool", "web_search"]):
        logger.info("All tools have been used, ending conversation")
        return "end"
    
    # If the last message is a function response, continue
    if hasattr(last_message, "name") and last_message.name in tool_belt:
        logger.info(f"Received response from {last_message.name}, continuing")
        return "continue"
    
    # If the last message has a function call, continue
    if (
        hasattr(last_message, "additional_kwargs") and 
        "function_call" in last_message.additional_kwargs
    ):
        logger.info("Message contains function call, continuing to execute it")
        return "continue"
    
    # Default to continuing if unsure
    return "continue"


def dummy_node(state):
    return


def get_state_update_bot_with_helpfullness_node():
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tool)
    workflow.add_node("helpfulness", check_helpfulness)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "tools": "tools",
            "helpfulness": "helpfulness"
        }
    )
    
    # Add edge from tools to helpfulness
    workflow.add_edge("tools", "helpfulness")
    
    # Add conditional edge from helpfulness
    workflow.add_conditional_edges(
        "helpfulness",
        lambda x: x.get("next", "continue"),
        {
            "continue": "agent",
            "tools": "tools",
            "end": END
        }
    )
    
    # Don't use a checkpointer to avoid the configurable issue
    return workflow.compile(debug=True)

# Helper function to determine if tool should be used
def should_use_tool(state):
    if not state["messages"]:
        return "helpfulness"
        
    last_message = state["messages"][-1]
    
    # Check for function calls in various formats
    has_function_call = False
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "additional_kwargs"):
            if "function_call" in last_message.additional_kwargs:
                has_function_call = True
            elif "tool_calls" in last_message.additional_kwargs:
                has_function_call = True
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            has_function_call = True
            
    if has_function_call:
        logger.info("Detected tool call in message, routing to tools node")
        return "tools"
        
    return "helpfulness"

# Fixed check_helpfulness function
async def check_helpfulness(state):
    # Don't add messages to the state here
    if len(state["messages"]) > 20:
        logger.info("Message count exceeded maximum, ending conversation")
        return {"next": "end"}
    
    last_message = state["messages"][-1]
    
    # Check for function calls in various formats
    has_function_call = False
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "additional_kwargs"):
            if "function_call" in last_message.additional_kwargs:
                has_function_call = True
            elif "tool_calls" in last_message.additional_kwargs:
                has_function_call = True
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            has_function_call = True
    
    # If we have a function call, route to tools
    if has_function_call:
        logger.info("Detected tool call, routing to tools node")
        return {"next": "tools"}
    
    # If we have a final AI response with content but no function call, end
    if (isinstance(last_message, AIMessage) and 
        hasattr(last_message, "content") and
        last_message.content and
        not has_function_call):
        logger.info("Detected final AI response, ending conversation")
        return {"next": "end"}
    
    # If we have a function response, continue to agent
    if hasattr(last_message, "name") and last_message.name in ["pubmed_search", "rag_tool", "web_search", "error", "timeout", "arxiv_search", "git_info"]:
        logger.info(f"Received response from {last_message.name}, continuing to agent")
        return {"next": "agent"}
    
    # Default to continuing
    logger.info("No special conditions detected, continuing to agent")
    return {"next": "continue"}

def convert_inputs(input_object):
    system_prompt = f"""You are a qualified psychologist providing mental health advice. Be empathetic in your responses. 
    Always provide a complete response. Be empathetic and provide a follow-up question to find a resolution. 
    
    You have access to several tools:
    1. pubmed_search: Search PubMed for medical research articles
    2. rag_tool: Search the knowledge base for mental health information
    3. web_search: Search the web for information 
    4. git_info: Get information about a git repository
    
    use tools at your dsiposal. Make sure to consult with rag_tool and pubmed_search and web_search.
    
    User's question: {input_object["messages"]}
    """
    return {"messages": [SystemMessage(content=system_prompt)]}

def extract_references(tool_results):
    """Extract and format references from tool results based on their source."""
    references = []
    
    # Extract PubMed references
    if "pubmed_search" in tool_results:
        content = tool_results["pubmed_search"]
        if not ("No results" in content or "Error" in content):
            # Try to extract PubMed references using common patterns
            
            # Look for citations in format: Author, A. A., et al. (Year). Title. Journal, Volume(Issue), pages.
            citation_pattern = re.compile(r'([A-Za-z\s,\.]+et al\.\s+\(\d{4}\)\.\s+[^\.]+\.[^\.]+\.)')
            citations = citation_pattern.findall(content)
            
            # If no citations found with that pattern, look for titles and authors separately
            if not citations:
                # Look for paper titles
                title_pattern = re.compile(r'"([^"]+)"')
                titles = title_pattern.findall(content)
                
                # Look for PMID references
                pmid_pattern = re.compile(r'PMID:?\s*(\d+)')
                pmids = pmid_pattern.findall(content)
                
                # Combine into formatted citations
                for i, title in enumerate(titles[:min(3, len(titles))]):
                    pmid = pmids[i] if i < len(pmids) else "N/A"
                    citations.append(f"{title}. PMID: {pmid}")
            
            # Add formatted citations
            for citation in citations[:min(3, len(citations))]:
                references.append(f"[PubMed] {citation}")
    
    # Extract arXiv references
    if "arxiv_search" in tool_results:
        content = tool_results["arxiv_search"]
        if not ("No results" in content or "Error" in content):
            # Try to extract arXiv references using common patterns
            
            # Look for arXiv IDs
            arxiv_pattern = re.compile(r'arXiv:(\d+\.\d+)')
            arxiv_ids = arxiv_pattern.findall(content)
            
            # Look for paper titles
            title_pattern = re.compile(r'"([^"]+)"')
            titles = title_pattern.findall(content)
            
            # Combine into formatted citations
            for i, title in enumerate(titles[:min(3, len(titles))]):
                arxiv_id = arxiv_ids[i] if i < len(arxiv_ids) else "N/A"
                references.append(f"[arXiv] {title}. arXiv:{arxiv_id}")
    
    # Extract web search references
    if "web_search" in tool_results:
        content = tool_results["web_search"]
        if not ("No results" in content or "Error" in content):
            # Try to extract URLs
            
            # Look for URLs
            url_pattern = re.compile(r'https?://[^\s\)]+')
            urls = url_pattern.findall(content)
            
            # Add formatted web references
            for url in urls[:min(3, len(urls))]:
                # Clean up URL and truncate if too long
                clean_url = url.strip(',.)')
                if len(clean_url) > 60:
                    clean_url = clean_url[:57] + "..."
                references.append(f"[Web] {clean_url}")
    
    # Return references
    return references

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    message: str
    references: list = []
    session_id: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize state if this is a new session
        if session_id not in active_sessions:
            active_sessions[session_id] = {"messages": [], "tool_results": {}}
        
        session = active_sessions[session_id]
        
        # For simplicity, let's just use sequential tool execution instead of the graph
        # This avoids issues with the LangGraph configuration
        
        # Initial human message
        human_message = HumanMessage(content=request.message)
        messages = [
            SystemMessage(content="""You are a qualified psychologist providing mental health advice.
            Be empathetic and thoughtful in your responses."""),
            human_message
        ]
        
        # Get previous messages if any
        if session["messages"]:
            messages = session["messages"] + [human_message]
        
        # Tool results storage
        tool_results = {}
        
        # SEQUENTIAL TOOL EXECUTION - No graph needed
        # 1. First execute pubmed_search
        logger.info("Searching PubMed for scientific research...")
        try:
            pubmed_tool = next(t for t in tool_belt if t.name == "pubmed_search")
            
            # Execute with timeout
            start_time = time.time()
            pubmed_result = await asyncio.wait_for(
                asyncio.to_thread(pubmed_tool.func, request.message), 
                timeout=45.0
            )
            
            # Log success
            execution_time = time.time() - start_time
            logger.info(f"Tool pubmed_search completed in {execution_time:.2f} seconds")
            
            # Store result
            tool_results["pubmed_search"] = pubmed_result
            
            # Add result to messages for the model
            messages.append(FunctionMessage(content=pubmed_result, name="pubmed_search"))
            
        except asyncio.TimeoutError:
            logger.warning("Tool pubmed_search timed out after 45 seconds")
            pubmed_result = "Timeout: PubMed search took too long. Using general knowledge instead."
            tool_results["pubmed_search"] = pubmed_result
            messages.append(FunctionMessage(content=pubmed_result, name="pubmed_search"))
            
        except Exception as e:
            logger.error(f"Tool pubmed_search failed: {str(e)}")
            pubmed_result = f"Error in PubMed search: {str(e)}. Using general knowledge instead."
            tool_results["pubmed_search"] = pubmed_result
            messages.append(FunctionMessage(content=pubmed_result, name="pubmed_search"))
        
        # 2. Then execute rag_tool
        logger.info("Searching knowledge base...")
        try:
            rag_tool = next(t for t in tool_belt if t.name == "rag_tool")
            
            # Execute with timeout
            start_time = time.time()
            rag_result = await asyncio.wait_for(
                asyncio.to_thread(rag_tool.func, request.message), 
                timeout=45.0
            )
            
            # Log success
            execution_time = time.time() - start_time
            logger.info(f"Tool rag_tool completed in {execution_time:.2f} seconds")
            
            # Store result
            tool_results["rag_tool"] = rag_result
            
            # Add to messages
            messages.append(FunctionMessage(content=rag_result, name="rag_tool"))
                    
        except asyncio.TimeoutError:
            logger.warning("Tool rag_tool timed out after 45 seconds")
            rag_result = "Timeout: Knowledge base search took too long. Using general knowledge instead."
            tool_results["rag_tool"] = rag_result
            messages.append(FunctionMessage(content=rag_result, name="rag_tool"))
            
        except Exception as e:
            logger.error(f"Tool rag_tool failed: {str(e)}")
            rag_result = f"Error in knowledge base search: {str(e)}. Using general knowledge instead."
            tool_results["rag_tool"] = rag_result
            messages.append(FunctionMessage(content=rag_result, name="rag_tool"))
        
        # 3. Finally execute web_search
        logger.info("Searching the web for current information...")
        try:
            web_tool = next(t for t in tool_belt if t.name == "web_search")
            
            # Execute with timeout
            start_time = time.time()
            web_result = await asyncio.wait_for(
                asyncio.to_thread(web_tool.func, request.message), 
                timeout=45.0
            )
            
            # Log success
            execution_time = time.time() - start_time
            logger.info(f"Tool web_search completed in {execution_time:.2f} seconds")
            
            # Store result
            tool_results["web_search"] = web_result
            
            # Add to messages
            messages.append(FunctionMessage(content=web_result, name="web_search"))
            
        except asyncio.TimeoutError:
            logger.warning("Tool web_search timed out after 45 seconds")
            web_result = "Timeout: Web search took too long. Using general knowledge instead."
            tool_results["web_search"] = web_result
            messages.append(FunctionMessage(content=web_result, name="web_search"))
                
        except Exception as e:
            logger.error(f"Tool web_search failed: {str(e)}")
            web_result = f"Error in web search: {str(e)}. Using general knowledge instead."
            tool_results["web_search"] = web_result
            messages.append(FunctionMessage(content=web_result, name="web_search"))
        
        # Add final instruction to model
        messages.append(SystemMessage(
            content="""Based on all the information gathered, provide a comprehensive, empathetic response 
            to the user's question. Include insights from scientific research, knowledge base, and current 
            web information as appropriate. Focus on practical, actionable advice."""
        ))
        
        logger.info("Generating final response...")
        
        # Get final response from the model
        response = await model.ainvoke(messages)
        final_response = response.content
        
        # Update session with new messages
        session["messages"] = messages + [AIMessage(content=final_response)]
        session["tool_results"] = tool_results
        
        # Extract references
        references = extract_references(tool_results)
        
        return ChatResponse(
            message=final_response,
            references=references,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Main function to run the server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Create a session store for tracking active conversations
active_sessions = {}

if __name__ == "__main__":
    run_server()