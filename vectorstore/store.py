from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
import pickle

VECTORSTORE_DIR = "./vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.chmod(VECTORSTORE_DIR, 0o777)

def get_vectorstore_path(project_name: str) -> str:
    return os.path.join(VECTORSTORE_DIR, f"{project_name}.pkl")

def build_vectorstore(chunks, project_name: str):
    docs = [
        Document(page_content=chunk, metadata={"project": project_name})
        for chunk in chunks
    ]

    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings()
    )
    
    # Save the vectorstore
    vectorstore_path = get_vectorstore_path(project_name)
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    return vectorstore

def project_exists(project_name: str) -> bool:
    vectorstore_path = get_vectorstore_path(project_name)
    return os.path.exists(vectorstore_path)

def get_retriever(project_name: str):
    if not project_exists(project_name):
        raise ValueError(f"Project '{project_name}' does not exist")
        
    # Load the vectorstore
    vectorstore_path = get_vectorstore_path(project_name)
    with open(vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"project": project_name}}
    )
    return retriever
