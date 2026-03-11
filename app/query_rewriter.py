from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def create_query_rewriter():

    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens=64,
        return_full_text=False,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    def rewrite(question):

        prompt = f"""
Rewrite the following question so it is clear and specific for searching a document.

Question:
{question}

Rewritten Question:
"""

        result = llm.invoke(prompt)

        return result.strip()

    return rewrite