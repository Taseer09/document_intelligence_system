from app.loader import load_pdf
from app.splitter import split_docs
from app.vector_store import create_vector_store
from app.qa import create_qa

print("Loading document...")
documents = load_pdf("data/sample.pdf")

print("Splitting document...")
chunks = split_docs(documents)

print("Creating vector store...")
vector_store = create_vector_store(chunks)

print("Building QA system...")
qa = create_qa(vector_store)

print("\nSystem Ready! 🚀 Ask questions about the document.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Question: ")

    if question.strip().lower() in {"exit", "quit"}:
        print("Goodbye.")
        break

    if not question.strip():
        continue
    
    print("Thinking...")

    # 1. Pass the question using the "query" key
    result = qa.invoke({"query": question})

    print("\nANSWER:")
    # 2. Extract the AI's response using the "result" key
    print(result["result"])

    print("\nRETRIEVED CONTEXT:")
    # 3. Extract the chunks using the "source_documents" key
    for i, doc in enumerate(result.get("source_documents", [])):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content[:300] + "...")