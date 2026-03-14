# Document Intelligence System

Document Intelligence System is a local Retrieval-Augmented Generation (RAG) application for querying PDF documents through a FastAPI backend and a Streamlit frontend. It is designed for private document analysis, grounded answers, and a straightforward local deployment workflow.

## Highlights

- Local-first PDF question answering workflow.
- Hybrid retrieval using FAISS and BM25.
- Cross-encoder reranking for better context selection.
- Grounded response generation with a constrained QA prompt.
- Streamlit UI for upload and chat.
- Docker Compose support for running API and UI together.

## Overview

The pipeline currently does the following:

- Loads PDF files from an uploaded document or the local `data/` folder.
- Splits documents into chunks with `RecursiveCharacterTextSplitter`.
- Builds a FAISS vector index using `BAAI/bge-large-en-v1.5` embeddings.
- Retrieves candidates with both FAISS and BM25.
- Re-ranks retrieved chunks with `BAAI/bge-reranker-base`.
- Generates grounded answers with `Qwen/Qwen2.5-1.5B-Instruct`.

The main supported interface is:

- FastAPI backend: `app/api.py`
- Streamlit UI: `app/ui.py`

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| LLM Runtime | Hugging Face Transformers, LangChain HuggingFace |
| Generation Model | Qwen/Qwen2.5-1.5B-Instruct |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Retrieval | FAISS, BM25 |
| Reranking | BAAI/bge-reranker-base |
| Document Processing | PyPDF, LangChain text splitters |
| ML Dependencies | Torch, Sentence Transformers, Accelerate |
| Containerization | Docker, Docker Compose |

## Hardware Requirements

The project can run on CPU, but practical performance depends heavily on available memory and accelerator support.

### Minimum

- OS: Windows 10 or Windows 11
- Python: 3.10
- CPU: Modern 4-core processor
- RAM: 16 GB
- Storage: 10 GB free space for dependencies and downloaded models
- GPU: Not required, but expect slow inference on CPU-only setups

### Recommended

- OS: Windows 10 or Windows 11
- Python: 3.10 in a Conda environment
- CPU: 6-core or better
- RAM: 32 GB
- GPU: NVIDIA GPU with 8 GB or more VRAM
- Storage: SSD with at least 20 GB free space for models, caches, and containers

### Notes

- First run requires internet access to download Hugging Face models.
- The generation pipeline uses `device_map="auto"`, so execution placement depends on detected hardware.
- Dense retrieval, reranking, and generation all add memory pressure, especially for larger PDFs.

## Project Structure

```text
document_intelligence_system/
|-- app/
|   |-- api.py
|   |-- bm25_retriever.py
|   |-- loader.py
|   |-- qa.py
|   |-- splitter.py
|   |-- ui.py
|   |-- vector_store.py
|   `-- evaluation/
|-- data/
|-- frontend/
|-- Dockerfile
|-- docker-compose.yml
|-- main.py
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.10
- `pip` or Conda
- Internet access on first run to download model artifacts
- Sufficient RAM or VRAM for embeddings, reranking, and answer generation

## Installation

### Option 1: Conda on Windows

```powershell
conda create -n ai310 python=3.10 -y
conda activate ai310
pip install -r requirements.txt
```

### Option 2: Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

Start the API in one terminal:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Start the Streamlit UI in a second terminal:

```powershell
streamlit run app/ui.py
```

Then open:

- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

## Running With Docker Compose

Build and start both services:

```powershell
docker compose up --build
```

This starts:

- API on port `8000`
- Streamlit UI on port `8501`

The Compose file sets `API_URL=http://api:8000` for the UI container, so the two services can communicate inside Docker.

## How To Use

1. Open the Streamlit UI.
2. Upload a PDF file.
3. Wait for processing to finish.
4. Ask questions about the uploaded document.

You can also use the API directly.

### Upload a document

```bash
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf"
```

### Ask a question

```bash
curl -X POST "http://localhost:8000/chat" \
	-H "Content-Type: application/json" \
	-d '{"question":"What is this document about?"}'
```

## Current Behavior and Limitations

- Uploaded files are saved into `data/`.
- The vector store is built in memory each time documents are processed.
- Session memory in the API is short-lived and process-local.
- The repository includes older prototype files such as `frontend/chat_app.py`; the maintained Streamlit entry point is `app/ui.py`.
- `main.py` exists, but the actively supported user flow is the FastAPI plus Streamlit path documented above.

## Troubleshooting

- If model downloads fail, verify Hugging Face access from the machine or container.
- If the UI cannot connect, confirm the API is running on port `8000`.
- If processing is very slow, check whether the runtime is falling back to CPU.
- If Docker builds are large or slow, that is expected because model and ML dependencies are substantial.

## Development Notes

- Backend app object: `app.api:app`
- Streamlit app entry point: `app/ui.py`
- Docker uses the same image for both API and UI services

If you plan to extend the project, the first improvements worth making are persistent vector storage, better error handling around uploads, and a cleaned-up separation between active code and older prototype files.

