from huggingface_hub import InferenceClient
from engine.ocr import OCRProcessor
from engine.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 llm_model_name=None, 
                 embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 tesseract_path=None):
        
        # Load model name from environment if not provided
        if llm_model_name is None:
            llm_model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        
        self.ocr = OCRProcessor(tesseract_path)
        self.vector_store = VectorStore(model_name=embedding_model_name)
        
        # Initialize Inference Client
        # On HF Spaces, if HF_TOKEN secret is set, it's available as an env var.
        self.hf_token = os.getenv("HF_TOKEN")
        
        if not self.hf_token:
            print("WARNING: HF_TOKEN not found in environment. Using default Space permissions.")
        else:
            print(f"INFO: HF_TOKEN detected (starts with: {self.hf_token[:4]}...)")
            
        print(f"Initializing Inference API for natively-hosted model: {llm_model_name}...")
        # We don't specify the token if it's not found, as the Space might have a scoped token already.
        self.client = InferenceClient(model=llm_model_name, token=self.hf_token)
        print("RAG Pipeline initialized (Native API mode).")

    def process_document(self, content_bytes, filename):
        """Extract text, chunk it, and add to vector store."""
        text = self.ocr.extract_text_from_bytes(content_bytes, filename)
        if text.startswith("Error"):
            return text
        
        # Simple intelligent chunking (by paragraphs/sentences)
        chunks = self._chunk_text(text)
        self.vector_store.add_text_chunks(chunks)
        return f"Processed {len(chunks)} chunks from {filename}."

    def _chunk_text(self, text, chunk_size=700, overlap=100):
        """Split text into chunks with overlap using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "।", ".", " ", ""]
        )
        return text_splitter.split_text(text)

    def answer_query(self, query):
        """Retrieve context and generate answer."""
        context_chunks = self.vector_store.search(query, top_k=3)
        if not context_chunks:
            return "Context er moddhe ei file er kono tottho nai (No relevant context found).", []
        
        context = "\n\n".join(context_chunks)
        
        # Standard Chat Message Format for chat_completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question strictly using the provided context. Respond in the same language/script as the user."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        try:
            # Use chat_completion (the 'conversational' task)
            response = self.client.chat_completion(
                messages,
                max_tokens=512,
                temperature=0.3, # Low for precision
                top_p=0.9
            )
            
            clean_answer = response.choices[0].message.content.strip()
            
            if not clean_answer:
                return "I'm sorry, I couldn't generate a clear answer. Please try rephrasing.", context_chunks
            
            return clean_answer, context_chunks
        except Exception as e:
            # Detailed diagnostics
            import traceback
            print(f"CHAT ERROR:\n{traceback.format_exc()}")
            return f"Generation Error: {str(e)}", context_chunks
