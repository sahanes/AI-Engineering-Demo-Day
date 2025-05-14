"""
Direct RAG Implementation

This module provides a direct implementation of the RAG (Retrieval Augmented Generation)
functionality without relying on the server communication.
"""

from langchain_community.document_loaders import ArxivLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from operator import itemgetter
import os
import logging
import time
import httpx
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_rag")

# Load environment variables
load_dotenv()

class DirectRAG:
    """Direct implementation of RAG functionality."""
    
    def __init__(self, load_documents=True):
        """Initialize the DirectRAG with documents and vectorstore."""
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=30,
            max_retries=3,
            http_client=httpx.Client(verify=False)  # SSL verification disabled for troubleshooting
        )
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            request_timeout=30,
            max_retries=3,
            http_client=httpx.Client(verify=False)
        )
        
        # Create RAG prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are a professional mental health advisor. Use the following context to answer the user's query.
        If you cannot answer the question based on the context, use your own knowledge but make it clear
        that you're doing so.

        Context:
        {context}

        Question:
        {question}
        """)
        
        # Set up vectorstore and chain
        if load_documents:
            self._load_documents_and_create_vectorstore()
            self._create_chain()
    
    def _load_documents_and_create_vectorstore(self):
        """Load documents and create vectorstore."""
        logger.info("Loading documents from ArXiv...")
        start_time = time.time()
        
        try:
            # Load documents from ArXiv
            documents = ArxivLoader(
                query='"mental health" AND (anxiety OR "sleep problems" OR stress)',
                load_max_docs=10,
                sort_by="submittedDate",
                sort_order="descending",
                load_all_available_meta=True,
                download_pdf=False
            ).load()
            
            logger.info(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            logger.info("Created FAISS vectorstore")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            # Create an empty vectorstore if there's an error
            self.vectorstore = FAISS.from_texts(
                ["Mental health is the state of well-being. Anxiety and depression are common issues."], 
                self.embeddings
            )
    
    def _create_chain(self):
        """Create the RAG chain."""
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create chain
        self.chain = (
            {"context": itemgetter("query") | self.retriever, "question": itemgetter("query")}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, query_str):
        """Run a query through the RAG system."""
        logger.info(f"Processing query: {query_str}")
        start_time = time.time()
        
        try:
            # Run the query
            response = self.chain.invoke({"query": query_str})
            logger.info(f"Query processed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your query: {str(e)}"

# Singleton instance
_rag_instance = None

def get_rag_instance():
    """Get or create a singleton RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = DirectRAG()
    return _rag_instance

def direct_rag_query(query_str):
    """Direct function to query the RAG system."""
    rag = get_rag_instance()
    return rag.query(query_str)

# Test code
if __name__ == "__main__":
    # Test the RAG system
    query = "What are evidence-based approaches for dealing with anxiety and sleep problems?"
    result = direct_rag_query(query)
    print("\nQUERY:", query)
    print("\nRESULT:", result) 