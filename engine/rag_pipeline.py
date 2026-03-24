from huggingface_hub import InferenceClient
from engine.ocr import OCRProcessor
from engine.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 llm_model_name="google/gemma-2-2b-it", 
                 embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 tesseract_path=None):
        
        self.ocr = OCRProcessor(tesseract_path)
        self.vector_store = VectorStore(model_name=embedding_model_name)
        
        # Initialize Inference Client
        self.hf_token = os.getenv("HF_TOKEN")
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
        
        # Gemma-2 Chat Template
        prompt = f"""<start_of_turn>user
Use the provided context to answer the question. 
Respond in the same language/script as the question (Bengali, Banglish, or English).

Context:
{context}

Question: {query}<end_of_turn>
<start_of_turn>model
Answer:"""
        
        try:
            # Enhanced generation parameters
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                stop_sequences=["<end_of_turn>", "User:", "Question:"]
            )
            
            clean_answer = response.strip()
            if not clean_answer:
                return "I'm sorry, I couldn't generate a clear answer from the document context. Please try rephrasing or check the 'View Sources' section.", context_chunks
            
            # Remove any trailing "Answer:" prefixes if the model repeats them
            if clean_answer.startswith("Answer:"):
                clean_answer = clean_answer.replace("Answer:", "", 1).strip()
                
            return clean_answer, context_chunks
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown API Error"
            return f"Generation Error: {error_msg}. (Make sure your HF_TOKEN is correctly set in Settings > Secrets)", context_chunks
