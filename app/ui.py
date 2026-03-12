import streamlit as st
import requests

st.title("AI Document Assistant")

question = st.text_input("Ask a question")

if st.button("Ask"):
    
    with st.spinner("Thinking..."):
        # 1. Send the request
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}  # Note: Check if your backend expects "query" instead of "question"!
        )

        # 2. Check if the backend gave us a green light (Status 200 OK)
        if response.status_code == 200:
            try:
                answer = response.json()["answer"]
                st.write(answer)
            except KeyError:
                st.error("The API succeeded, but it didn't return an 'answer' key. Check your backend return dictionary!")
        
        # 3. If the backend rejected it or crashed, print exactly WHY!
        else:
            st.error(f"⚠️ API Error {response.status_code}")
            st.code(response.text) # This will print the actual error from FastAPI!