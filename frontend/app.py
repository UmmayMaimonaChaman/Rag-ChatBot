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

def get_system_status():
    """Check if the backend is online and ready."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json().get("status", "unknown")
        return "down"
    except:
        return "down"

def upload_file(file):
    files = {"file": file}
    try:
        response = requests.post(f"{API_URL}/upload", files=files, timeout=600)
        res_json = response.json()
        if response.status_code != 200 or "error" in res_json:
            error_msg = res_json.get("error") or res_json.get("detail") or "Unknown server error"
            return {"error": error_msg}
        return res_json
    except Exception as e:
        return {"error": f"Connection failed. The system might still be starting up. (Error: {str(e)})"}

def ask_query(query):
    try:
        response = requests.post(f"{API_URL}/query", json={"query": query}, timeout=300)
        res_json = response.json()
        if response.status_code != 200 or "error" in res_json:
            error_msg = res_json.get("error") or res_json.get("detail") or "Generation failed"
            return {"error": error_msg}
        return res_json
    except Exception as e:
        return {"error": f"Query failed. API connection error: {str(e)}"}

def clear_data():
    try:
        requests.post(f"{API_URL}/clear", timeout=5)
        st.session_state.messages = []
        return True
    except:
        return False

# --- UI Layout ---
# Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://img.icons8.com/fluency/96/bot.png", width=60)
with col2:
    st.markdown('<div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-bottom: 0;">RAG ChatBot</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #94a3b8; font-size: 1rem; margin-top: -10px;">Multilingual (Bengali-Banglish-English) Document Intelligence System</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;">Control Center</div>', unsafe_allow_html=True)
    
    # Status Indicator
    status = get_system_status()
    if status == "ready":
        st.success("🟢 System Online & Ready")
    elif status == "loading":
        st.warning("🟡 System Initializing (Downloading Models...)")
        st.info("This can take 2-5 minutes on the first run.")
    else:
        st.error("🔴 System Offline (Backend Starting...)")
        if st.button("🔄 Refresh Status"):
            st.rerun()

    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg"], help="Upload PDF or Image for analysis")
    
    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True, disabled=(status != "ready")):
            with st.status("Processing document...", expanded=True) as status_box:
                st.write("Extracting content and generating embeddings...")
                res = upload_file(uploaded_file)
                if "error" in res:
                    st.error(f"{res['error']}")
                    status_box.update(label="Process failed!", state="error")
                else:
                    st.success("Document analyzed and indexed!")
                    status_box.update(label="Document ready!", state="complete")
                    st.balloons()
    
    st.divider()
    if st.button("🗑️ Clear Context", use_container_width=True):
        if clear_data():
            st.toast("Chat history cleared!")
            time.sleep(0.5)
            st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Discussion area
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("📚 View Sources"):
                    for i, chunk in enumerate(message["chunks"]):
                        st.markdown(f"**Chunk {i+1}:**\n{chunk}")

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            res = ask_query(prompt)
            if "error" in res:
                response = f"⚠️ Error: {res['error']}"
                chunks = []
            else:
                response = res.get("answer", "No answer received.")
                chunks = res.get("chunks", [])
            
            st.markdown(response)
            if chunks:
                with st.expander("📚 View Sources"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}:**\n{chunk}")
                        
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "chunks": chunks
            })

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">A simple chatbot made by Ummay Maimona Chaman as a first learning outcome of RAG</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">© All Rights Reserved to Ummay Maimona Chaman</div>', unsafe_allow_html=True)
