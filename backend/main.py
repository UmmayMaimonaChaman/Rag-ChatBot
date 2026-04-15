from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import sys
from pydantic import BaseModel
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import engine
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine.rag_pipeline import RAGPipeline

app = FastAPI(title="Multilingual RAG Chatbot API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
rag = None

@app.get("/")
async def root():
    return {"status": "online"}

@app.get("/health")
async def health():
    if rag:
        return {"status": "ready"}
    return {"status": "loading"}

@app.on_event("startup")
async def startup_event():
    global rag
    try:
        logger.info("Starting RAG Pipeline initialization...")
        # Using high-performance Inference API models
        rag = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize RAG Pipeline: {str(e)}")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    chunks: List[str]
    status: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not rag:
        return {"error": "System is still initializing embedding models. Please wait a moment."}
    
    try:
        logger.info(f"Uploading file: {file.filename}")
        content = await file.read()
        result = rag.process_document(content, file.filename)
        
        if result.startswith("Error"):
            logger.error(f"Processing failed for {file.filename}: {result}")
            return {"error": result}
            
        logger.info(f"File {file.filename} processed successfully.")
        return {"message": result}
    except Exception as e:
        logger.exception(f"Unexpected error during upload of {file.filename}")
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    if not rag:
        return {"error": "System is still initializing. Please try again in a few seconds."}
        
    try:
        logger.info(f"Received query: {request.query}")
        answer, chunks = rag.answer_query(request.query)
        return QueryResponse(answer=answer, chunks=chunks, status="success")
    except Exception as e:
        logger.exception("Error during query processing")
        return {"error": str(e), "answer": "An error occurred while generating the answer.", "chunks": [], "status": "error"}

@app.post("/clear")
async def clear_store():
    try:
        if rag:
            rag.vector_store.clear()
            logger.info("Vector store cleared.")
            return {"message": "Vector store cleared."}
        return {"error": "System not initialized."}
    except Exception as e:
        logger.error(f"Error clearing store: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
