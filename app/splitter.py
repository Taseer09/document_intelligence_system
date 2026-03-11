from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks