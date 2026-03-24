from huggingface_hub import InferenceClient
from engine.ocr import OCRProcessor
from engine.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, 
                 llm_model_name="mistralai/Mistral-7B-Instruct-v0.2", 
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
            return "No relevant context found in the uploaded documents."
        
        context = "\n\n".join(context_chunks)
        
        prompt = f"""
        You are a Document Intelligence Assistant. Use the provided context to answer the user's question.
        
        SUPPORTED LANGUAGES:
        - English
        - Bengali (বাংলা)
        - Banglish (Bengali language written using the Latin/English alphabet, e.g., "Kemon acho?")
        
        CORE INSTRUCTIONS:
        1. Process Banglish (Latin-script Bengali) queries as Bengali.
        2. Respond in the same language and script as the user's question.
        3. If the user asks in Banglish, reply in Banglish. 
        4. If the context is missing info, state: "Context er moddhe ei file er kono tottho nai" (for Banglish/Bengali queries).
        5. Stay strictly within the provided context.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        try:
            # Use InferenceClient for generation
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop_sequences=["Question:", "\n\n"]
            )
            return response.strip()
        except Exception as e:
            return f"Error during generation: {str(e)}"
