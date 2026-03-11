from langchain_community.document_loaders import PyPDFLoader

path = "D:\AI\AI Data\document_intelligence_system\data\sample.pdf"

def load_pdf(path):

    loader = PyPDFLoader(path)
    documents = loader.load()

    return documents