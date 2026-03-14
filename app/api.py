from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import shutil

# These are your custom modules - ensure these files exist in the app/ folder!
from app.qa import create_qa
from app.loader import load_documents
from app.splitter import split_docs
from app.vector_store import create_vector_store

app = FastAPI()

# Ensure the data folder exists locally
os.makedirs("data/", exist_ok=True)

# Global variables for the AI engine
vector_store = None
qa = None
session_memory = []

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store, qa
    
    try:
        # 1. Save the file locally
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Re-process the brain
        documents = load_documents("data/")
        chunks = split_docs(documents)
        vector_store = create_vector_store(chunks)
        qa = create_qa(vector_store, chunks)
        
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_answer(question: str):
    global session_memory, qa
    
    if qa is None:
        yield "Error: Please upload a document first."
        return

    # Basic memory management
    history_text = "\n".join(session_memory)
    full_query = f"Context:\n{history_text}\n\nQuestion: {question}" if session_memory else question

    # Run the model (This might be slow on CPU!)
    result = qa.invoke({"query": full_query})
    answer = result["result"]

    session_memory.append(f"User: {question}")
    session_memory.append(f"AI: {answer}")
    if len(session_memory) > 4: session_memory = session_memory[-4:]

    # Simulate streaming effect
    for word in answer.split(" "):
        yield word + " "
        await asyncio.sleep(0.05)

@app.post("/chat")
async def chat(q: Question):
    if qa is None:
         raise HTTPException(status_code=400, detail="No document uploaded")
    return StreamingResponse(stream_answer(q.question), media_type="text/plain")