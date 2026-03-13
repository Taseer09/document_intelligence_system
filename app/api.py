from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.qa import create_qa
from app.loader import load_documents
from app.splitter import split_docs
from app.vector_store import create_vector_store


app = FastAPI()

documents = load_documents("data/")
chunks = split_docs(documents)

vector_store = create_vector_store(chunks)

qa = create_qa(vector_store, chunks)


class Question(BaseModel):
    question: str


def stream_answer(question):

    result = qa.invoke({"query": question})

    answer = result["result"]

    for word in answer.split():
        yield word + " "


@app.post("/chat")

def chat(q: Question):

    return StreamingResponse(
        stream_answer(q.question),
        media_type="text/plain"
    )