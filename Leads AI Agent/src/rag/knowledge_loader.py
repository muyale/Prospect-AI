from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_knowledge_docs(directory: str = "knowledge/docs"):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            docs.extend(loader.load())
    return docs

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

if __name__ == "__main__":
    docs = load_knowledge_docs()
    vector_store = create_vector_store(docs)
    print("Knowledge base loaded into FAISS vector store.")
