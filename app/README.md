# 📄 Local AI Document Intelligence System

A completely offline, privacy-first Retrieval-Augmented Generation (RAG) system that allows users to securely chat with PDF documents. Built with LangChain, Hugging Face, and Streamlit, this application is heavily optimized to run locally on consumer-grade GPUs (specifically designed to fit within 6GB of VRAM).

## ✨ Features

* **100% Local & Private:** No API keys required. No data ever leaves your machine, making it perfect for sensitive or confidential documents.
* **Strict Factual Grounding:** Engineered with highly constrained, "paranoid" prompt templates to prevent the AI from hallucinating or using outside knowledge. It only answers based on the provided text.
* **Source Citations:** The chat interface provides expandable dropdowns showing the exact document chunks the AI used to generate its answer.
* **Advanced Retrieval:** Utilizes Maximum Marginal Relevance (MMR) to fetch diverse and highly relevant context chunks from the vector database.
* **Interactive Web UI:** Features a sleek, modern chat interface built with Streamlit.
* **CLI Fallback:** Includes a pure terminal-based loop (`main.py`) for quick debugging and headless testing.

## 🛠️ Tech Stack

* **Framework:** LangChain (`RetrievalQA` chains)
* **Frontend UI:** Streamlit
* **LLM Engine:** Hugging Face `transformers` & `langchain-huggingface`
* **Language Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Chosen for its balance of instruction-following and low memory footprint)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` 

## 💻 Hardware Requirements

This project has been optimized to run locally on mid-range hardware.
* **OS:** Windows 10/11
* **GPU:** NVIDIA GPU with at least **6GB VRAM** (e.g., RTX 3060)
* **RAM:** 16GB System RAM minimum
* **Python:** 3.10+ (Conda environment recommended)

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/document_intelligence_system.git](https://github.com/yourusername/document_intelligence_system.git)
cd document_intelligence_system