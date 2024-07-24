import json
import operator
from operator import itemgetter
from typing import Annotated, Sequence, TypedDict

import chainlit as cl
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain.storage import InMemoryStore

# from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun

# from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- #
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""

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


docs = ArxivLoader(
    query='"mental health counseling" AND (data OR analytics OR "machine learning")',
    load_max_docs=10,
    sort_by="submittedDate",
    sort_order="descending",
).load()


### 2. CREATE QDRANT CLIENT VECTORE STORE

client = QdrantClient(":memory:")
client.create_collection(
    collection_name="split_parents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vectorstore = Qdrant(
    client,
    collection_name="split_parents",
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
)

store = InMemoryStore()

### 3. CREATE PARENT DOCUMENT TEXT SPLITTER AND RETRIEVER INITIATED

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
parent_document_retriever.add_documents(docs)

### 4. CREATE PROMPT OBJECT
RAG_PROMPT = """\
Your are a professional mental helth advisor. Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

### 5. CREATE CHAIN PIPLINE RETRIVER

openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)


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


@tool
async def rag_tool(question: str) -> str:
    """Only use this tool to retrieve research relevant information from the knowledge base."""
    # advanced_rag_prompt=ChatPromptTemplate.from_template(INSTRUCTION_PROMPT_TEMPLATE.format(user_query=question))
    parent_document_retriever_qa_chain = create_qa_chain(parent_document_retriever)
    response = await parent_document_retriever_qa_chain.ainvoke({"question": question})

    return response["response"]


tool_belt = [
    rag_tool,
    PubmedQueryRun(),
    ArxivQueryRun(),
    DuckDuckGoSearchRun(),
]

tool_executor = ToolExecutor(tool_belt)


### 7. CONVERT TOOLS INTO THE FORMAT COMAPTIBLE WITH OPENAI'S FUNCTION CALLING API THEN BINDING THEM TO MODEL TO BE USED WHEN GENERATION
model = ChatOpenAI(temperature=0, streaming=True)

functions = [convert_to_openai_function(t) for t in tool_belt]
model = model.bind_functions(functions)
model = model.with_config(tags=["final_node"])

### 8. USING the TypedDict FROM THE typing module AND THE langchain_core.messages module, A CUSTOM TYPE NAMED AgentState CREATED.
# THE AgentState type HAS A FIELD NAMED <messages> THAT IS OF TYPE Annotated[Sequence[BaseMessage], operator.add].
# Sequence[BaseMessage]: INDICATES THAT MESSAGES ARE A SEQUENCE OF BaseMessage OBJECTS.
# Annotated: USED TO ATTACH MEATADATA TO THE TYPE, THEN THE MESSAGE FIELD TREATED AS CONCATENABLE SEQUENCE OF BASEMASSAGES TO OPERATOR.ADD FUNCTION.


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


### 9. TWO FUNCTIONS DEFINED: 1. call_model AND 2. call_tool FUNCTIONS
# 1. INVOKES THE MODEL BY THE MESSAGES EXTRACTED FROM THE STATE RETURNING A DICT CONTAINING THE RESPONSE MESSAGE,
# 2.1 ToolInvocation OBJECT CREATED USING THE NAME AND ARGUMENTS EXTRACTED FROM THE LAST MASSAGE EXTRACTED FROM THE STATE,
# 2.2. tool_executor IS INVOKED BY THE CREATED toolInvocation OBJECT
# 2.3 FunctionMessage OBJECT IS CREATED WITH THE tool_executor RESPONSE AND THE NAME OF THAT TOOL
# 2.4 RETURN IS A DICT CONTAINING FunctionMessage OBJECT.


async def call_model(state):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def call_tool(state):
    last_message = state["messages"][-1]

    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    print()
    print(last_message.additional_kwargs["function_call"]["name"])
    print()

    response = await tool_executor.ainvoke(action)

    function_message = FunctionMessage(content=str(response), name=action.tool)

    return {"messages": [function_message]}


###10. GRAPG CREATION WITH HELPFULNESS EVALUATION
# should_continue CHECKS IF THE LAST MASSAGE IN THE STATE IS TO CONTINUE (additional_kwargs EXISTS) OR END.
# THE add_conditional_edges() method IS ORIGINATED FROM THIS REPONSE, EITHER TRANSITION TO ACTION NODE OR END.


def should_continue(state):
    last_message = state["messages"][-1]

    if "function_call" not in last_message.additional_kwargs:
        return "end"

    return "continue"


async def check_helpfulness(state):
    initial_query = state["messages"][0]
    final_response = state["messages"][-1]

    # adding artificial_loop

    if len(state["messages"]) > 10:
        return "END"

    prompt_template = """\
  Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y'\
  and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

    prompt_template = PromptTemplate.from_template(prompt_template)

    helpfulness_check_model = ChatOpenAI(model="gpt-4")

    helpfulness_check_chain = (
        prompt_template | helpfulness_check_model | StrOutputParser()
    )

    helpfulness_response = await helpfulness_check_chain.ainvoke(
        {"initial_query": initial_query, "final_response": final_response}
    )

    if "Y" in helpfulness_response:
        print("helpful!")
        return "end"

    else:
        print(" Not helpful!!")
        return "continue"


def dummy_node(state):
    return


### 11. SETTING THE GRAPH WORKFLOW:
# 1. AN INSTANCE OF THE STATEGRAPH CREATED OF THE TYPE AgentState. THREE NODES ADDED TO THE GRAPH USING add_node() method:
# 1.1 THE "agent" NODE IS ASSOCIATED WITH THE call_model FUNCTION.
# 1.2 THE "action" NODE IS ASSOCIATED WITH THE call_tool FUNCTION.
# 1.3 THE "passthrough" NODE IS A CUSTOM NODE THAT IS ASSOCIATED WITH CHECKING HELPFULNESS.
# 1.5 THE CONDITIONAL EDGES
# 1.5.1 BETWEEN agent NODE AND THE OTHER TWO NODES TO EITHER action NODE OR passthrough NODE
# 1.5.2 BETWEEN passthrough NODE AND agen NODE OR END NODE.
# 1.5.3 BETWEEN agent AND action NODES AS MODEL HAS ACCESS TO TOOLS FOR RESPONSE GENERATION.
def get_state_update_bot():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)  # agent node has access to llm
    workflow.add_node("action", call_tool)  # action node has access to tools
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",  # tools
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")  # tools
    state_update_bot = workflow.compile()

    return state_update_bot


#   --------------------------------------------------
def get_state_update_bot_with_helpfullness_node():
    graph_with_helpfulness_check = StateGraph(AgentState)

    graph_with_helpfulness_check.add_node("agent", call_model)
    graph_with_helpfulness_check.add_node("action", call_tool)
    graph_with_helpfulness_check.add_node("passthrough", dummy_node)

    graph_with_helpfulness_check.set_entry_point("agent")

    graph_with_helpfulness_check.add_conditional_edges(
        "agent", should_continue, {"continue": "action", "end": "passthrough"}
    )

    graph_with_helpfulness_check.add_conditional_edges(
        "passthrough", check_helpfulness, {"continue": "agent", "end": END}
    )

    graph_with_helpfulness_check.add_edge("action", "agent")

    return graph_with_helpfulness_check.compile()


### 12.
# def convert_inputs(input_object):
#     system_prompt = f"""You are a qualified psychologist providing mental health advice. Be empathetic in your responses. 
#     Always provide a complete response. Be empathetic and provide a follow-up question to find a resolution. 
    

# You will operate in a loop of Thought, Action, PAUSE, and Observation. At the end of the loop, you will provide an Answer.

# Instructions:

# Thought: Describe your thoughts about the user's question.
# Action: Choose one of the available actions to gather information or provide insights.
# PAUSE: Pause to allow the action to complete.
# Observation: Review the results of the action.

# Available Actions:

# Use the tools at your disposal to look up information or resolve the consultancy. You are allowed to make multiple calls (either together or in sequence).:

# 1. pub_med: Find a specific coping strategies or management techniques by doing research paper
# 2. rag_tool: RAG (Retrieval-Augmented Generation) to access relevant mental health information.
# 3. duckduckgo_search: Perform an online search: InternetSearch to find up-to-date resources and recommendations.
# 4. arxiv: Find relevant research or content.

# You may make multiple calls to these tools as needed to provide comprehensive advice.

# Present your final response in a clear, structured format, including a chart of recommended actions if appropriate.

#     User's question: {input_object[-1]}

#     Response: Your task is When responding to users' personal issues or concerns:

# 1. With a brief empathetic acknowledgment of the user's situation, continue
# 2. Provide practical, actionable advice that often includes 
# 3. Suggesting professional help (e.g., therapists, counselors) when appropriate
# 4. Encouraging open communication and dialogue with involved parties and 
# 5. Recommending self-reflection or exploration of emotions and values and
# 6. Offering specific coping strategies or management techniques
# """
#     return {"messages": [SystemMessage(content=system_prompt)]}
def convert_inputs(input_object):
    system_prompt = f"""You are a qualified psychologist providing mental health advice. Be empathetic in your responses. 
    Always provide a complete response. Be empathetic and provide a follow-up question to find a resolution. 
    
    You must Use the tools at your dsiposal.
    You must consult pubmed, then ragtool, then duckduckgo_results_json.
    You must make multiple calls to these tools as needed to provide comprehensive advice.


    User's question: {input_object[-1]}
    """
    return {"messages": [SystemMessage(content=system_prompt)]}



# Define the function to parse the output
def parse_output(input_state):
    return input_state


# bot_with_helpfulness_check=get_state_update_bot_with_helpfullness_node() # type:
# bot=get_state_update_bot()

# Create the agent chain
# agent_chain = convert_inputs | bot_with_helpfulness_check# | StrOutputParser()#| parse_output

# Run the agent chain with the input
# messages=agent_chain.invoke({"question": mental_health_counseling_data['test'][14]['Context']})

# ---------------------------------------------------------------------------------------------------------
#                                       DEPLOYMENT
# ---------------------------------------------------------------------------------------------------------


@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message.

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {"Assistant": "Mental Health Advisor Bot"}
    return rename_dict.get(original_author, original_author)


@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session.

    We will build our LCEL RAG chain here, and store it in the user session.

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    # lcel_rag_chain = ( {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}

    #                    | rag_prompt | hf_llm
    #                 )

    bot_with_helpfulness_check = get_state_update_bot_with_helpfullness_node()  # type: ignore
    lcel_agent_chain = (
        convert_inputs | bot_with_helpfulness_check
    )  # StrOutputParser())/

    # bot=get_state_update_bot()

    # lcel_agent_chain = convert_inputs | bot| parse_output# StrOutputParser()

    cl.user_session.set("lcel_agent_chain", lcel_agent_chain)


# @cl.on_message
# async def main(message: cl.Message):
#     """
#     This function will be called every time a message is recieved from a session.

#     We will use the LCEL agent chain to generate a response to the user query.

#     The LCEL agent chain is stored in the user session, and is unique to each user session - this is why we can access it here.
#     """
#     lcel_agent_chain = cl.user_session.get("lcel_agent_chain")

#     msg = cl.Message(content="")
    
#     async for event in lcel_agent_chain.astream_events(
#         [message.content],
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#         version="v2",
#     ):
#         kind = event["event"]
#         tags = event.get("tags", [])
        
#     #     if kind == "on_chat_model_stream" and "final_node" in tags:
#     #         data = event["data"]
#     #         if data["chunk"].content:
#     #             await msg.stream_token(data["chunk"].content)

#     # await msg.send()
#         print({event})
#         if kind == "on_chain_stream":# and "tool_end" in tags:
#             data = event["data"]#['output']
#             print(data)
#             if data["chunk"]:#.content:
#                 print(f"Streaming content: {data['chunk']}")#.content}")  
#                 await msg.stream_token(data["chunk"])#.content)
                
#         await msg.send()

# chainlit run app.py -w

#e isn't violent, but he has anger issues and deep insecurities. He's working on them and has improved. We started counseling, and he participated in one or two individual sessions, but we broke up again shortly thereafter. Now his constant questions and accusations are getting really draining.
@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL agent chain to generate a response to the user query.

    The LCEL agent chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_agent_chain = cl.user_session.get("lcel_agent_chain")

    msg = cl.Message(content="")
    # await msg.send()
    final_output=""

    async for event in lcel_agent_chain.astream_events(
        [message.content],
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        version="v2",
    ):
        kind = event["event"]
        tags = event.get("tags", [])
        name=event.get("name", "")
        print(f"Received event: {event}")  # Debugging statement

        if kind == "on_chain_end" and name=="RunnableSequence":#"tool_end" in tags:
            if 'output' in event['data'] and "agent" in event["data"]['output']:
                agent_output=event["data"]["output"]["agent"]
                if "messages" in agent_output and agent_output["messages"]:
                    final_output=agent_output["messages"][0].content
                    await msg.stream_token(final_output)

        elif kind=="on_chat_model_stream":
            data=event['data']
            if data["chunk"].content:
                print(f"Streaming content: {data['chunk'].content}")  
                await msg.stream_token(data["chunk"].content)


    await msg.send()
