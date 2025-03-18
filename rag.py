# rag.py
import os
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from llm import get_llm

logger = logging.getLogger(__name__)

# Global variables
vector_store = None
retriever_chain = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_index_document(file_bytes: bytes, api_key: str = None) -> str:
    """
    Reads and indexes the uploaded document into FAISS.
    If no API key is provided, attempts to load it from the OPENAI_API_KEY environment variable.
    """
    global vector_store, retriever_chain

    # If no API key provided, attempt to retrieve it from environment variable.
    if not api_key:
        api_key ="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"
        if not api_key:
            raise ValueError("API key is required. Provide it as a parameter or set the OPENAI_API_KEY environment variable.")
    
    # Convert file bytes to text
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("The uploaded document is empty or unreadable.")

    # Split text into chunks
    docs = text_splitter.create_documents([text])
    if not docs:
        raise ValueError("No valid text found in the uploaded file.")

    # Initialize OpenAI embeddings
    logger.info("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # FAISS does not support dynamic updates, so we create a new vector store
    logger.info("Creating new FAISS vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Create a retriever and RetrievalQA chain
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = get_llm(api_key)
    retriever_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    logger.info("Document indexed successfully!")
    return "Document indexed successfully."

from typing import Tuple, List

def get_answer(query: str) -> Tuple[str, List[str]]:
    """
    Uses the RAG pipeline to retrieve and generate an answer.
    """
    global retriever_chain
    if retriever_chain is None:
        raise ValueError("No document has been indexed yet. Please upload a document first.")

    # Query the retrieval chain
    result = retriever_chain({"query": query})
    
    # Extract answer and source documents
    answer = result.get("result", "No answer found.")
    source_docs = result.get("source_documents", [])
    sources = [doc.metadata.get("source", "N/A") for doc in source_docs]

    return answer, sources