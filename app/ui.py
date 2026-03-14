import streamlit as st
import requests
import os

# --- DYNAMIC CONFIGURATION ---
# This looks for 'API_URL' in the system. 
# If it's not found (local run), it defaults to localhost.
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Doc Intel SaaS", page_icon="🚀", layout="centered")

# --- SESSION STATE ---
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

def reset_app():
    st.session_state.is_ready = False
    st.session_state.messages = []
    st.session_state.active_doc = None
    st.rerun()

# --- HEADER ---
st.title("🚀 Document Intelligence SaaS")

if st.session_state.is_ready:
    cols = st.columns([0.7, 0.3])
    with cols[0]:
        st.success(f"📄 **Active:** {st.session_state.active_doc}")
    with cols[1]:
        if st.button("Switch Document", use_container_width=True):
            reset_app()
    st.divider()

# --- STATE 1: UPLOADER ---
if not st.session_state.is_ready:
    st.markdown("#### Upload a document to begin.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                # Uses the dynamic URL
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                if response.status_code == 200:
                    st.session_state.is_ready = True
                    st.session_state.active_doc = uploaded_file.name
                    st.rerun()
                else:
                    st.error("Backend error during processing.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# --- STATE 2: CHAT ---
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask about {st.session_state.active_doc}..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                # Uses the dynamic URL
                resp = requests.post(f"{API_BASE_URL}/chat", json={"question": prompt}, stream=True)
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except:
                st.error("API connection lost.")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})