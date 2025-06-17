from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os

CHROMA_DIR = "./chroma_store"
os.makedirs(CHROMA_DIR, exist_ok=True)
os.chmod(CHROMA_DIR, 0o777)  

def build_vectorstore(chunks, project_name: str):
    docs = [
        Document(page_content=chunk, metadata={"project": project_name})
        for chunk in chunks
    ]

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DIR
    )
    return vectorstore

def project_exists(project_name: str) -> bool:
    try:
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_DIR
        )
        results = vectorstore.get(
            where={"project": project_name},
            limit=1
        )
        return len(results['ids']) > 0
    except Exception:
        return False

def get_retriever(project_name: str):
    if not project_exists(project_name):
        raise ValueError(f"Project '{project_name}' does not exist")
        
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_DIR
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"project": project_name}}
    )
    return retriever
