import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

class VectorStore:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', index_path='vector_store/faiss_index'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        
        if os.path.exists(f"{self.index_path}.bin"):
            self.load_index()

    def add_text_chunks(self, chunks):
        """Convert chunks to embeddings and add to FAISS index."""
        if not chunks:
            return
        
        embeddings = self.model.encode(chunks)
        self.index.add(np.array(embeddings).astype('float32'))
        self.chunks.extend(chunks)
        self.save_index()

    def search(self, query, top_k=5):
        """Search for relevant chunks given a query."""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = [self.chunks[idx] for idx in indices[0] if idx != -1]
        return results

    def save_index(self):
        """Save index and chunks to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.bin")
        with open(f"{self.index_path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        """Load index and chunks from disk."""
        if os.path.exists(f"{self.index_path}.bin"):
            self.index = faiss.read_index(f"{self.index_path}.bin")
            with open(f"{self.index_path}_chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)

    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        if os.path.exists(f"{self.index_path}.bin"):
            os.remove(f"{self.index_path}.bin")
        if os.path.exists(f"{self.index_path}_chunks.pkl"):
            os.remove(f"{self.index_path}_chunks.pkl")
