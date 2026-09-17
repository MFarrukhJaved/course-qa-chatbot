"""
TeachRAG - Embeddings

Local sentence-embedding model via sentence-transformers. No API calls,
no student material leaves the machine this runs on - same "100% local"
principle as the ChemALLM project, just applied to text embeddings
instead of a compound database.
"""

from typing import List, Optional
import numpy as np

from core.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    """Lazy-load the sentence-transformers model (it's a ~90MB download on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings -> (N, dim) float32 array, L2-normalized."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
