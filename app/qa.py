from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

from app.bm25_retriever import BM25Retriever
from sentence_transformers import CrossEncoder


def create_qa(vector_store, chunks):

    # ---------------------------
    # PROMPT
    # ---------------------------

    template = """<|im_start|>system
You are a strict document assistant. Your ONLY source of information is the Context below.
Read the Context carefully. You are allowed to combine details from the Context to answer the question.
If the answer cannot be found in the Context, you MUST output exactly: "I cannot answer this based on the provided document."
Do NOT use outside knowledge.

Context:
{context}<|im_end|>

<|im_start|>user
Question: {question}
Remember: Only use the Context.<|im_end|>

<|im_start|>assistant
"""

    qa_prompt = PromptTemplate.from_template(template)

    # ---------------------------
    # LLM
    # ---------------------------

    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
        max_length=None,
        return_full_text=False,
        model_kwargs={"device_map": "auto"},
        do_sample=False,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # ---------------------------
    # VECTOR RETRIEVER
    # ---------------------------

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    # ---------------------------
    # BM25 RETRIEVER
    # ---------------------------

    bm25 = BM25Retriever(chunks)

    # ---------------------------
    # RERANKER
    # ---------------------------

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # ---------------------------
    # HYBRID RETRIEVAL LOGIC
    # ---------------------------

    def hybrid_retrieve(query):

        vector_docs = vector_retriever.invoke(query) # Updated to .invoke() to avoid deprecation warnings!

        bm25_docs = bm25.retrieve(query)

        combined = vector_docs + bm25_docs

        # remove duplicates
        unique_docs = list({doc.page_content: doc for doc in combined}.values())

        # reranking
        pairs = [(query, doc.page_content) for doc in unique_docs]

        scores = reranker.predict(pairs)

        scored_docs = list(zip(scores, unique_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs = [doc for _, doc in scored_docs[:4]]

        return top_docs

    # ---------------------------
    # CUSTOM RETRIEVER WRAPPER (FIXED)
    # ---------------------------

    # Inherit from BaseRetriever to pass LangChain's validation!
    class CustomHybridRetriever(BaseRetriever):
        
        # You must use this exact function name with the underscore
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return hybrid_retrieve(query)

    # Instantiate the new class
    hybrid_retriever_instance = CustomHybridRetriever()

    # ---------------------------
    # QA CHAIN
    # ---------------------------

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever_instance, # <--- Passes the bouncer!
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa