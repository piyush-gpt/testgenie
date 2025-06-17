from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
import json

VECTORSTORE_DIR = "./vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.chmod(VECTORSTORE_DIR, 0o777)

def get_vectorstore_path(project_name: str) -> str:
    return os.path.join(VECTORSTORE_DIR, project_name)

def build_vectorstore(chunks, project_name: str):
    docs = [
        Document(page_content=chunk, metadata={"project": project_name})
        for chunk in chunks
    ]

    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings()
    )
    
    # Save the vectorstore using FAISS's native save method
    vectorstore_path = get_vectorstore_path(project_name)
    vectorstore.save_local(vectorstore_path)
    
    return vectorstore

def project_exists(project_name: str) -> bool:
    vectorstore_path = get_vectorstore_path(project_name)
    return os.path.exists(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "index.faiss"))

def get_retriever(project_name: str):
    if not project_exists(project_name):
        raise ValueError(f"Project '{project_name}' does not exist")
        
    # Load the vectorstore using FAISS's native load method
    vectorstore_path = get_vectorstore_path(project_name)
    vectorstore = FAISS.load_local(
        vectorstore_path,
        OpenAIEmbeddings()
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"project": project_name}}
    )
    return retriever
