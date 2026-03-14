import streamlit as st
import requests

st.set_page_config(page_title="Document AI", page_icon="📚", layout="centered")

# --- INITIALIZE STATE ---
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- THE HEADER ---
st.title("📚 Document Intelligence")
st.markdown("Upload a document, and the AI will become an expert on it instantly.")
st.divider()

# --- STATE 1: THE UPLOAD ZONE (Centered & Professional) ---
if not st.session_state.is_ready:
    st.markdown("### Step 1: Initialize the Engine")
    uploaded_file = st.file_uploader("Drag and drop your PDF here", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        with st.spinner("Processing document and spinning up neural networks..."):
            # Send the file to our new FastAPI endpoint
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post("http://api:8000/upload", files=files)
            
            if response.status_code == 200:
                st.session_state.is_ready = True
                st.rerun() # Refresh the screen to hide the uploader!
            else:
                st.error("Failed to process the document. Please try again.")

# --- STATE 2: THE CHAT INTERFACE ---
else:
    # 1. Draw all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Handle new user input
    if prompt := st.chat_input("Ask a question about your document..."):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                response = requests.post(
                    "http://api:8000/chat", 
                    json={"question": prompt},
                    stream=True
                )
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ API Connection Error: {e}")
                full_response = "Sorry, I couldn't reach the AI engine."

        st.session_state.messages.append({"role": "assistant", "content": full_response})