from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from pydantic import BaseModel
from typing import List

from pathlib import Path

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

@app.on_event("startup")
async def startup_event():
    global rag
    # Using high-performance Inference API models
    rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    status: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = rag.process_document(content, file.filename)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    try:
        answer = rag.answer_query(request.query)
        return QueryResponse(answer=answer, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_store():
    try:
        rag.vector_store.clear()
        return {"message": "Vector store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
