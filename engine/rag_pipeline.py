from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from engine.ocr import OCRProcessor
from engine.vector_store import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class RAGPipeline:
    def __init__(self, 
                 llm_model_name="microsoft/phi-2", 
                 embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 tesseract_path=None):
        
        self.ocr = OCRProcessor(tesseract_path)
        self.vector_store = VectorStore(model_name=embedding_model_name)
        
        # Initialize LLM
        print(f"Loading LLM: {llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            torch_dtype=torch.float32, 
            device_map="auto",
            trust_remote_code=True
        )
        self.generator = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        print("LLM Loaded successfully.")

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
        You are a Document Intelligence Assistant. Use the following context to answer the user's question. 
        If you don't know the answer based on the context, say that the context does not contain the information.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        response = self.generator(prompt)
        # Extract response after the prompt
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        return answer
