from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate

def create_qa(vector_store):

    # 1. Keep the better reading comprehension prompt
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

    # 2. Roll back to the 1.5B Model
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=256,
        max_length=None, # Keeps the warnings away
        return_full_text=False,
        model_kwargs={"device_map": "auto"}, # Loads natively to your 6GB VRAM
        do_sample=False,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    # 3. Keep the "stuff" chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )

    return qa