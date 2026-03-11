from langchain_community.vectorstores import FAISS
from app.embeddings import load_embeddings

def create_vector_store(chunks):

    embeddings = load_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store