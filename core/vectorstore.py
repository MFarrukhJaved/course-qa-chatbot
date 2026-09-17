"""
TeachRAG - Vector Store

A small FAISS-backed index over the chunk embeddings, persisted to disk
so the class material doesn't need to be re-embedded every time the app
restarts. Local file-based storage only - no external vector DB service.
"""

import json
import os
from typing import List, Dict, Tuple

import numpy as np
import faiss

from core.config import INDEX_PATH, METADATA_PATH


class VectorStore:
    def __init__(self):
        self.index: faiss.Index = None
        self.chunks: List[Dict] = []  # parallel to index vectors, by row order

    # --------------------------------------------------
    # Build / update
    # --------------------------------------------------
    def build(self, chunks: List[Dict], vectors: np.ndarray):
        """Build a fresh index from scratch (replaces any existing index)."""
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine, since vectors are normalized
        if len(vectors) > 0:
            self.index.add(vectors)
        self.chunks = chunks

    def add(self, chunks: List[Dict], vectors: np.ndarray):
        """Append new chunks/vectors to an existing index."""
        if self.index is None:
            self.build(chunks, vectors)
            return
        if len(vectors) > 0:
            self.index.add(vectors)
        self.chunks.extend(chunks)

    # --------------------------------------------------
    # Search
    # --------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[Dict, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = query_vector.reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def save(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    def load(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH) -> bool:
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            return False
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        return True

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def stats(self) -> Dict:
        sources = sorted({c["source"] for c in self.chunks})
        return {
            "num_chunks": len(self.chunks),
            "num_documents": len(sources),
            "documents": sources,
        }
