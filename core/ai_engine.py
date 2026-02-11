import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import MAX_SEQ_LENGTH, CHUNK_OVERLAP_TOKENS

class AIEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_token_chunks(self, text):
        """True Token-Aware Chunking logic."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), MAX_SEQ_LENGTH - CHUNK_OVERLAP_TOKENS):
            chunk_tokens = tokens[i : i + MAX_SEQ_LENGTH]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    def get_embedding(self, text, identifier):
        """Global Caching Layer."""
        cache_path = os.path.join(self.cache_dir, f"{identifier}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        
        emb = self.model.encode(text)
        np.save(cache_path, emb)
        return emb

    def compute_similarity(self, v1, v2):
        return float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0])