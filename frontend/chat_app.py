import sys
import os
import streamlit as st

# Force Python to look one folder up so it can find the 'app' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.loader import load_pdf
from app.splitter import split_docs
from app.vector_store import create_vector_store
from app.qa import create_qa

# ... [the rest of your Streamlit code stays exactly the same] ...

# 1. Page Configuration
st.set_page_config(page_title="AI Document Chat", page_icon="📄")
st.title("📄 AI Document Chat")

# 2. Initialize Session State Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa" not in st.session_state:
    st.session_state.qa = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# 3. Sidebar for File Upload (Keeps the chat area clean!)
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Check if this is a NEW file. If so, clear memory and re-process!
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.qa = None
            st.session_state.messages = []  # Clear old chat history

            # Use getbuffer() for safer Streamlit file saving
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document... This may take a moment."):
                docs = load_pdf("temp.pdf")
                chunks = split_docs(docs)
                vector_store = create_vector_store(chunks)
                
                st.session_state.qa = create_qa(vector_store)
                
            st.success(f"Processed {uploaded_file.name} successfully!")

# 4. Main Chat Interface
if st.session_state.qa is None:
    st.info("👈 Please upload a PDF document in the sidebar to start chatting!")
else:
    # Display the existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If the AI remembered sources for this message, display them!
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for i, doc in enumerate(msg["sources"]):
                        st.write(f"**Source {i+1}**")
                        st.write(doc.page_content[:300] + "...")

    # 5. Handle New User Questions
    prompt = st.chat_input("Ask something about the document...")

    if prompt:
        # Show user prompt immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and show AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ask the QA system
                    result = st.session_state.qa.invoke({"query": prompt})
                    answer = result.get("result", "I could not generate an answer.")
                    sources = result.get("source_documents", [])

                    # Print the answer
                    st.markdown(answer)

                    # Print the sources
                    if sources:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(sources):
                                st.write(f"**Source {i+1}**")
                                st.write(doc.page_content[:300] + "...")

                    # Save EVERYTHING (including sources) to memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")