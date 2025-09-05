from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from typing import List

class Embedder:
    def __init__(self, model_name: str = "models/bge-small-en", db_path: str = "db"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.index_file = os.path.join(self.db_path, "faiss.index")
        self.meta_file = os.path.join(self.db_path, "metas.pkl")
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.metas: List[str] = []

    def _to_float32_2d(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def encode_and_store(self, texts: List[str]):
        vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vectors = self._to_float32_2d(vectors)
        self.index.add(vectors)
        self.metas.extend(texts)
        self.save()

    def save(self):
        os.makedirs(self.db_path, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metas, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "rb") as f:
                self.metas = pickle.load(f)
        if hasattr(self.index, "ntotal") and len(self.metas) != self.index.ntotal:
            self.metas = self.metas[: self.index.ntotal]

    def search(self, query: str, top_k: int = 5):
        if getattr(self.index, "ntotal", 0) == 0:
            return []
        q = self.model.encode([query], convert_to_numpy=True)
        q = self._to_float32_2d(q)
        D, I = self.index.search(q, top_k)
        return [self.metas[i] for i in I[0] if i != -1]

