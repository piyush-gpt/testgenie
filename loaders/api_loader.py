import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_openapi_spec(path: str) -> str:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return str(data)

def chunk_spec_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)
