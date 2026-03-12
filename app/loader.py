import os
from langchain_community.document_loaders import PyPDFLoader


def load_documents(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(os.path.join(folder_path, file))

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file

            documents.extend(docs)

    return documents