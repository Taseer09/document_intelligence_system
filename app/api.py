from fastapi import FastAPI
from pydantic import BaseModel

from app.qa import create_qa
from app.loader import load_documents
from app.splitter import split_docs
from app.vector_store import create_vector_store

# 1. Initialize FastAPI
app = FastAPI()

# 2. Load and Process Documents on Startup
documents = load_documents("data/")
chunks = split_docs(documents)
vector_store = create_vector_store(chunks)

# 3. Create the QA Chain (No internal LangChain memory)
qa = create_qa(vector_store, chunks)

# 4. Define the API Input Model
class Question(BaseModel):
    question: str

# ---------------------------------------------------------
# OUR CUSTOM MEMORY BANK (Global List)
# ---------------------------------------------------------
session_memory = []

@app.post("/ask")
def ask_question(q: Question):
    
    # Tell Python we are modifying the global list defined above!
    global session_memory

    # 1. Turn our memory list into a readable text block
    history_text = "\n".join(session_memory)
    
    # 2. Glue the history and the new question together
    if session_memory:
        full_query = f"Previous Conversation Context:\n{history_text}\n\nNew Question: {q.question}"
    else:
        full_query = q.question

    # 3. Ask LangChain (It only sees one string, so it won't crash!)
    result = qa.invoke({"query": full_query})
    answer = result["result"]

    # 4. Save this current interaction to our memory list for next time
    session_memory.append(f"User: {q.question}")
    session_memory.append(f"AI: {answer}")

    # 5. Keep only the last 4 messages so the GPU doesn't run out of memory!
    if len(session_memory) > 4:
        session_memory = session_memory[-4:]
        
    # 6. Return the answer to Streamlit
    return {
        "answer": answer
    }