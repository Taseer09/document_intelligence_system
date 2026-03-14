from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import shutil

from app.qa import create_qa
from app.loader import load_documents
from app.splitter import split_docs
from app.vector_store import create_vector_store

app = FastAPI()

# Make sure the data folder exists
os.makedirs("data/", exist_ok=True)

# Global variables to hold our AI brain
vector_store = None
qa = None
session_memory = []

class Question(BaseModel):
    question: str

# --- NEW: THE FILE UPLOAD RECEIVER ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store, qa
    
    # 1. Save the incoming PDF to our data folder
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Re-read the documents and update the AI's brain
    documents = load_documents("data/")
    chunks = split_docs(documents)
    vector_store = create_vector_store(chunks)
    qa = create_qa(vector_store, chunks)
    
    return {"message": "Document processed successfully"}

# --- UPDATED: THE CHAT GENERATOR ---
async def stream_answer(question: str):
    global session_memory, qa
    
    if qa is None:
        yield "Error: No document has been uploaded yet."
        return

    history_text = "\n".join(session_memory)
    if session_memory:
        full_query = f"Previous Conversation Context:\n{history_text}\n\nNew Question: {question}"
    else:
        full_query = question

    result = qa.invoke({"query": full_query})
    answer = result["result"]

    session_memory.append(f"User: {question}")
    session_memory.append(f"AI: {answer}")

    if len(session_memory) > 4:
        session_memory = session_memory[-4:]

    words = answer.split(" ")
    for i, word in enumerate(words):
        yield word + " " if i < len(words) - 1 else word 
        await asyncio.sleep(0.03)

@app.post("/chat")
async def chat(q: Question):
    if qa is None:
         raise HTTPException(status_code=400, detail="No document uploaded")
         
    return StreamingResponse(
        stream_answer(q.question),
        media_type="text/plain"
    )