import streamlit as st
import requests
import os
import time

# --- Page Config ---
st.set_page_config(
    page_title="Multilingual RAG Chatbot - Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & Styling ---
# Palette: #004771 (Primary), Dark Theme
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0e1117;
        color: #ffffff;
    }}
    .main-header {{
        color: #004771;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }}
    .sub-header {{
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: center;
    }}
    .stButton>button {{
        background-color: #004771;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #006096;
        box-shadow: 0 4px 12px rgba(0, 71, 113, 0.4);
    }}
    .chat-bubble {{
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        max-width: 85%;
    }}
    .user-bubble {{
        background-color: #1e293b;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }}
    .bot-bubble {{
        background-color: #004771;
        margin-right: auto;
        border-bottom-left-radius: 2px;
    }}
    /* Mobile optimization */
    @media (max-width: 768px) {{
        .main-header {{ font-size: 1.8rem; }}
        .chat-bubble {{ max-width: 95%; }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- Backend API URL ---
API_URL = "http://127.0.0.1:8000"

def check_backend():
    try:
        # Check if the root or a health endpoint works
        requests.get(API_URL, timeout=1)
        return True
    except:
        return False

def upload_file(file):
    files = {"file": file}
    try:
        response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        return response.json()
    except Exception as e:
        return {"error": f"Backend unreachable (might still be loading models). Original error: {str(e)}"}

def ask_query(query):
    try:
        response = requests.post(f"{API_URL}/query", json={"query": query}, timeout=90)
        return response.json()
    except Exception as e:
        return {"error": f"Backend connection error. Original error: {str(e)}"}

def clear_data():
    try:
        requests.post(f"{API_URL}/clear", timeout=5)
        st.session_state.messages = []
        return True
    except:
        return False

# --- UI Layout ---
st.markdown('<div class="main-header">🤖 Multilingual RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Bilingual (Bengali-English) Document Intelligence System</div>', unsafe_allow_html=True)

# Backend Status Check
if not check_backend():
    st.warning("⚠️ **AI Engine is still starting up...** (This can take 1-2 minutes on first load while models are being cached).")

# Sidebar for controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bot.png", width=80)
    st.header("Document Control")
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        if st.button("🚀 Process Document"):
            if not check_backend():
                st.error("Wait! The AI Engine is still loading. Please try again in a few seconds.")
            else:
                with st.spinner("Extracting text and building index..."):
                    res = upload_file(uploaded_file)
                    if "error" in res:
                        st.error(f"{res['error']}")
                    else:
                        st.success(res["message"])
    
    st.divider()
    if st.button("🗑️ Clear All History"):
        if clear_data():
            st.success("Cleared!")
            st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not check_backend():
            response = "⚠️ The AI Engine is still starting up. Please wait a moment and try again."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Thinking..."):
                res = ask_query(prompt)
            if "error" in res:
                response = f"⚠️ API Error: {res['error']}. Make sure the backend is running."
            else:
                response = res.get("answer", "No answer received.")
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("Powered by HuggingFace, FAISS and Tesseract OCR. Bilingual Bengali-English Support.")
