from huggingface_hub import InferenceClient
from engine.ocr import OCRProcessor
from engine.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 llm_model_name="HuggingFaceH4/zephyr-7b-beta", 
                 embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 tesseract_path=None):
        
        self.ocr = OCRProcessor(tesseract_path)
        self.vector_store = VectorStore(model_name=embedding_model_name)
        
        # Initialize Inference Client
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("CRITICAL: HF_TOKEN not found in environment!")
        else:
            print(f"HF_TOKEN detected (starts with: {self.hf_token[:4]}...)")
            
        print(f"Initializing Inference API for model: {llm_model_name}...")
        self.client = InferenceClient(model=llm_model_name, token=self.hf_token)
        print("RAG Pipeline initialized (API mode).")

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
        
        # Zephyr/Mistral Prompt Template
        prompt = f"""<|system|>
You are a helpful assistant. Answer the question using the context below. 
If the question is in Bengali or Banglish, answer in the same script/style.
Context:
{context}</s>
<|user|>
{query}</s>
<|assistant|>
"""
        
        try:
            # Enhanced generation parameters
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.3, # Low for precision
                top_p=0.9,
                repetition_penalty=1.1,
                stop_sequences=["</s>", "<|user|>", "<|system|>"]
            )
            
            clean_answer = response.strip()
            if not clean_answer:
                return "I'm sorry, I couldn't generate a clear answer from the document context. Please try rephrasing or check the 'View Sources' section.", context_chunks
            
            return clean_answer, context_chunks
        except Exception as e:
            # Capture more error details
            import traceback
            full_error = traceback.format_exc()
            print(f"GENERATION ERROR:\n{full_error}")
            error_msg = str(e) if str(e) else "Unknown API Error (Possibly rate limit or model timeout)"
            return f"Generation Error: {error_msg}. (Model: HuggingFaceH4/zephyr-7b-beta)", context_chunks
