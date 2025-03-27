
import os
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from llm import get_llm

logger = logging.getLogger(__name__)

vector_store = None
retriever_chain = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_index_document(file_bytes: bytes, api_key: str = None) -> str:
    global vector_store, retriever_chain

    if not api_key:
        api_key =""
        if not api_key:
            raise ValueError("API key is required. Provide it as a parameter or set the OPENAI_API_KEY environment variable.")
    
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("The uploaded document is empty or unreadable.")

    docs = text_splitter.create_documents([text])
    if not docs:
        raise ValueError("No valid text found in the uploaded file.")

    logger.info("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(api_key=api_key)

    logger.info("Creating new FAISS vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)

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
    global retriever_chain
    if retriever_chain is None:
        raise ValueError("No document has been indexed yet. Please upload a document first.")

    result = retriever_chain({"query": query})
    
    answer = result.get("result", "No answer found.")
    source_docs = result.get("source_documents", [])
    sources = [doc.metadata.get("source", "N/A") for doc in source_docs]

    return answer, sources
